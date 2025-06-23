import argparse
import time
from dataclasses import dataclass
from pathlib import Path
import logging
import queue
import threading
import io
import pickle
import sys
import tty
import termios
import select

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import audio
from just_playback import Playback
from openai import OpenAI, Timeout, APITimeoutError
import maths
from maths import scurve as sc, jcurve as jc, rcurve as rc, clamp01

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.logging import RichHandler
from rich.box import HEAVY
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn

# from pynput import keyboard


@dataclass
class TimedToken:
    token: str
    timestamp: float # in seconds
    loudness: float # from 0.0 to 1.0


@dataclass
class PlaybackState:
    playing: bool = True
    restart: bool = False
    quit: bool = False
    seek_by: int = 0  # in tokens
    seek_to_start: bool = False
    toggle_pause_request: bool = False


@dataclass
class AppConfig:
    """
    Application configuration.
    """
    song_path: Path
    prompt: str
    lm_studio_url: str = "http://localhost:1234/v1"
    model: str = "local-model"
    tps: int = 45 # tokens per second
    min_temp: float = 0.985
    max_temp: float = 1.875
    max_tokens: int = 50
    save_run_path: Path | None = None
    # This flag is to prevent __post_init__ from creating a new save path when loading a run
    _is_loaded_run: bool = False 

    def __post_init__(self):
        if self.save_run_path is None and not self._is_loaded_run:
            runs_dir = Path(".runs")
            runs_dir.mkdir(exist_ok=True)
            timestamp_str = time.strftime("%Y-%m-%d_%H-%M-%S")
            self.save_run_path = runs_dir / f"run_{timestamp_str}.npz"


@dataclass
class GenerationRun:
    """
    Holds all the data needed to replay a generation run.
    """
    config: AppConfig
    timed_tokens: list[TimedToken]
    temp: np.ndarray
    lufs: np.ndarray
    lufs_raw: np.ndarray
    v_lufs: np.ndarray

    def save(self, path: Path):
        """Saves the run data to a compressed .npz file."""
        # Deconstruct timed_tokens
        tokens = [t.token for t in self.timed_tokens]
        timestamps = [t.timestamp for t in self.timed_tokens]
        loudness_values = [t.loudness for t in self.timed_tokens]

        # Deconstruct config, converting Path to string for serialization
        config_dict = {
            'song_path': str(self.config.song_path),
            'prompt': self.config.prompt,
            'lm_studio_url': self.config.lm_studio_url,
            'model': self.config.model,
            'tps': self.config.tps,
            'min_temp': self.config.min_temp,
            'max_temp': self.config.max_temp,
            'max_tokens': self.config.max_tokens,
            'save_run_path': str(self.config.save_run_path) if self.config.save_run_path else 'None'
        }
        
        save_dict = {
            **config_dict,
            'tokens': np.array(tokens, dtype=object),
            'timestamps': np.array(timestamps),
            'loudness_values': np.array(loudness_values),
            'temp': self.temp,
            'lufs': self.lufs,
            'lufs_raw': self.lufs_raw,
            'v_lufs': self.v_lufs
        }
        
        np.savez_compressed(path, **save_dict)

    @classmethod
    def load(cls, path: Path) -> 'GenerationRun':
        """Loads a run from a .npz file."""
        with np.load(path, allow_pickle=True) as data:
            # Reconstruct config
            config = AppConfig(
                song_path=Path(str(data['song_path'])),
                prompt=str(data['prompt']),
                lm_studio_url=str(data['lm_studio_url']),
                model=str(data['model']),
                tps=int(data['tps']),
                min_temp=float(data['min_temp']),
                max_temp=float(data['max_temp']),
                max_tokens=int(data['max_tokens']),
                save_run_path=Path(str(data['save_run_path'])) if str(data['save_run_path']) != 'None' else None,
                _is_loaded_run=True
            )

            # Reconstruct timed_tokens
            timed_tokens = [
                TimedToken(token=t, timestamp=ts, loudness=l) 
                for t, ts, l in zip(data['tokens'], data['timestamps'], data['loudness_values'])
            ]

            return GenerationRun(
                config=config,
                timed_tokens=timed_tokens,
                temp=data['temp'],
                lufs=data['lufs'],
                lufs_raw=data['lufs_raw'],
                v_lufs=data['v_lufs']
            )


def colorize_token(token: str, loudness: float) -> str:
    """
    Colorizes a token based on loudness, using the viridis colormap on a black background.
    Loudness is expected to be in the [0, 1] range.
    """
    # Get RGB color from viridis colormap
    # The colormap returns a tuple (R, G, B, A) with values in [0, 1]
    cmap = cm.get_cmap('viridis')

    # Curve up the brightness in the low ends by applying a nonlinear mapping to the color itself,
    # not the loudness. We'll use a gamma correction (gamma < 1) on the RGB values after sampling
    # from the colormap, making the colors more vivid for low loudness.
    gamma = 0.75  # You can tweak this value for more/less boost
    rgb_float = cmap(loudness)[:3]
    rgb_curved = tuple(c ** gamma for c in rgb_float)
    r, g, b = [int(c * 255) for c in rgb_curved]

    # ANSI escape code for 24-bit color
    # Foreground color
    fg_color_code = f"\x1b[38;2;{r};{g};{b}m"
    # Background color (black)
    bg_color_code = "\x1b[48;2;0;0;0m"
    
    reset_code = "\x1b[0m"
    return f"{bg_color_code}{fg_color_code}{token}{reset_code}"


class GenerationTimeoutError(Exception):
    pass


def get_client(config: AppConfig) -> OpenAI:
    """
    Get OpenAI client for LM Studio.
    """
    # Set a 5 second read timeout. If the server doesn't send any data
    # for 5 seconds, it will raise an APITimeoutError.
    timeout = Timeout(5.0)
    return OpenAI(base_url=config.lm_studio_url, api_key="not-needed", timeout=timeout)


def get_token_worker(client, config, prompt, temperature, result_queue):
    """
    This function runs in a separate thread to isolate the blocking network call,
    which has proven to be an unstable, untrustworthy catastrophe that hangs
    for no reason. We tried client-side timeouts. We tried signals. All failed.
    The only remaining option is to run this garbage in a separate thread
    and hope we can kill it if it misbehaves.

    It requests a single token and puts the resulting chunk or an exception
    into the provided queue.
    """
    try:
        response = client.completions.create(
            model=config.model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=1,
            stream=True,
        )
        chunk = next(response)
        result_queue.put(chunk)
    except Exception as e:
        # If anything goes wrong in this house of cards, at least we can
        # pass the exception back to the main thread instead of just dying.
        result_queue.put(e)


def generate_all_tokens(
    client: OpenAI, 
    config: AppConfig, 
    temperature: np.ndarray, 
    live: Live,
    header_text: Text,
    generation_text: Text,
    dynamic_info_text: Text
) -> list[TimedToken]:
    """
    Generates all tokens from the LLM, with timestamps based on virtual time.
    """
    # Use a simple prompt format. The entire history is resent for each token
    # to allow for temperature changes, using the completions endpoint.
    prompt_prefix = f"User: {config.prompt}\nAssistant:"
    history = ""

    timed_tokens: list[TimedToken] = []
    token_index = 0
    
    # A safe value for context window to prevent hangs with local models
    MAX_HISTORY_CHARS = 2048

    header_text.plain = "Generating tokens..."
    live.refresh()

    while True:
        if token_index >= config.max_tokens:
            logging.info(f"Reached max tokens ({config.max_tokens}). Stopping generation.")
            break
            
        # Determine temperature for this token
        frame_index = min(token_index, len(temperature) - 1)
        temp = temperature[frame_index]
        
        current_time_s = token_index / config.tps
        dynamic_info_text.plain = (
            f"Token: {token_index + 1} / {config.max_tokens}\n"
            f"Time: {current_time_s:.2f}s\n"
            f"Temperature: {temp:.3f}"
        )
        
        # Derive the loudness value that was used to generate this temperature.
        # This is for storing in TimedToken and for colorization.
        loudness_value = (temp - config.min_temp) / (config.max_temp - config.min_temp)
        prompt = prompt_prefix + history

        try:
            result_queue = queue.Queue()
            worker = threading.Thread(
                target=get_token_worker,
                args=(client, config, prompt, temp, result_queue)
            )
            worker.daemon = True  # So it doesn't block app exit
            worker.start()

            # Here we wait for the worker thread to do its thing.
            # If the network call in the worker thread hangs forever, this timeout
            # is our final, desperate line of defense to prevent a soft-lock.
            # After trying every other "normal" way to handle this, we are left
            # with this monstrosity. Godspeed.
            result = result_queue.get(timeout=5.0)

            if isinstance(result, Exception):
                # The worker thread died, but at least it was kind enough
                # to tell us why. We re-raise the exception here to handle it.
                raise result

            chunk = result
            token = chunk.choices[0].text
            finish_reason = chunk.choices[0].finish_reason
            # logging.info(f"Received token: '{token}', finish_reason: {finish_reason}")
            
            colorized_token = colorize_token(token.replace('\n', ' '), loudness_value)
            generation_text.append(Text.from_ansi(colorized_token))
            live.refresh()

            timestamp = token_index / config.tps
            timed_tokens.append(TimedToken(token, timestamp, loudness_value))
            
            # Append the new token to the history and truncate if necessary
            history += token
            if len(history) > MAX_HISTORY_CHARS:
                history = history[-MAX_HISTORY_CHARS:]

            token_index += 1

            if finish_reason == "stop":
                logging.info("Finish reason 'stop' received, breaking loop.")
                break

        except queue.Empty:
            # The worker thread has gone silent. It's probably stuck in some
            # god-forsaken C library loop. We have no choice but to abandon it.
            logging.warning("WORKER THREAD HUNG. The API call timed out and the thread is now a zombie. Breaking generation.")
            break
        except Exception as e:
            logging.error(f"An unexpected error occurred in the generation loop: {e}")
            break
            
    header_text.plain = "Generation complete."
    live.refresh()
    return timed_tokens


def run_tui(
    config: AppConfig, 
    temp: np.ndarray, 
    lufs: np.ndarray, 
    lufs_raw: np.ndarray, 
    v_lufs: np.ndarray,
    timed_tokens: list[TimedToken] | None = None,
    is_loaded_run: bool = False
) -> list[TimedToken] | None:
    """
    Sets up and runs the Rich TUI for generation and playback.
    """
    console = Console()
    log_stream = io.StringIO()
    log_console = Console(file=log_stream, force_terminal=True, width=120)
    
    # We re-configure logging for each TUI run to reset the stream
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        datefmt="[%X]",
        handlers=[RichHandler(console=log_console, rich_tracebacks=True, show_path=False)],
        force=True 
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Layout
    layout = Layout()
    layout.split(
        Layout(name="header", size=5),
        Layout(name="main"),
        Layout(size=10, name="footer"),
    )

    layout["main"].split_row(
        Layout(name="main_content"),
        Layout(name="side_panel", size=40, minimum_size=35)
    )
    
    header_text = Text(f"Prompt: {config.prompt}\nSong: {config.song_path}", justify="center")
    generation_text = Text("", justify="left")
    playback_text = Text("", justify="left")
    
    layout["main_content"].update(Panel(generation_text, title="Generation", border_style="green", box=HEAVY))

    log_panel = Panel("", title="[bold cyan]Logs[/]", border_style="red", box=HEAVY)
    layout["footer"].update(log_panel)
    
    # Side Panel Setup
    song_duration = lufs_raw.shape[0] / config.tps
    lufs_raw_stats = f"{np.min(lufs_raw):.2f} -> {np.max(lufs_raw):.2f} (μ: {np.mean(lufs_raw):.2f})"
    lufs_proc_stats = f"{np.min(lufs):.2f} -> {np.max(lufs):.2f} (μ: {np.mean(lufs):.2f})"
    v_lufs_stats = f"{np.min(v_lufs):.2f} -> {np.max(v_lufs):.2f} (μ: {np.mean(v_lufs):.2f})"
    temp_stats = f"{np.min(temp):.2f} -> {np.max(temp):.2f} (μ: {np.mean(temp):.2f})"
    
    info_table = Table(show_header=False, box=None, padding=(0, 1))
    info_table.add_column(style="bold blue")
    info_table.add_column(style="white")
    info_table.add_row("Song Duration:", f"{song_duration:.2f}s")
    info_table.add_row("Tokens to Gen:", str(config.max_tokens))
    info_table.add_row("LUFS (raw):", lufs_raw_stats)
    info_table.add_row("LUFS (proc):", lufs_proc_stats)
    info_table.add_row("Velocity:", v_lufs_stats)
    info_table.add_row("Temperature:", temp_stats)

    dynamic_info_text = Text("", justify="left")
    info_group = Group(
        Panel(info_table, title="[bold yellow]Audio Analysis[/]", border_style="yellow", box=HEAVY),
        Panel(dynamic_info_text, title="[bold yellow]Live Stats[/]", border_style="yellow", box=HEAVY)
    )
    layout["side_panel"].update(info_group)


    layout["header"].update(
        Panel(header_text, title="[bold magenta]DJ-Token[/]", border_style="magenta", box=HEAVY)
    )

    # We need a separate console for the final print, otherwise it gets weird.
    final_console = Console()
    
    tokens = timed_tokens

    # Non-blocking input setup
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(sys.stdin.fileno())
        with Live(layout, console=console, screen=True, refresh_per_second=60, redirect_stderr=False, transient=True) as live:
            def update_logs():
                log_panel.renderable = Text.from_ansi(log_stream.getvalue())
                live.refresh()
            
            update_logs()

            # GENERATE if not provided
            # ----------------------------------------
            if not tokens:
                client = get_client(config)
                tokens = generate_all_tokens(client, config, temp, live, header_text, generation_text, dynamic_info_text)
                update_logs()
            
            if not tokens:
                logging.error("No tokens were generated. Exiting.")
                update_logs()
                return
                
            # PLAYBACK & INTERACTIVE LOOP
            # ----------------------------------------
            playstate = PlaybackState()

            header_text.plain = f"Playing '{config.song_path}'"
            layout["main_content"].update(Panel(playback_text, title="Playback", border_style="blue", box=HEAVY))
            
            # Command Bar
            pbar = Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            )
            pbar_task = pbar.add_task("Loading...", total=len(tokens))
            
            command_hints_text = "P/Space: Play/Pause | H: Home | ←/→: Seek | Q: Quit"
            if not is_loaded_run:
                command_hints_text = "R: Restart | " + command_hints_text
            command_hints = Text(command_hints_text, justify="center")
            command_bar_group = Group(pbar, command_hints)
            command_bar_panel = Panel(command_bar_group, title="[bold]Commands[/]", border_style="green", box=HEAVY)
            layout["footer"].split(Layout(log_panel), Layout(command_bar_panel, name="command_bar", size=5))

            live.refresh()
            
            playback = Playback()
            playback.load_file(str(config.song_path))
            playback.play()
            
            FPS = 60.0
            
            song_duration = lufs_raw.shape[0] / config.tps
            inext = 0
            t = 0.0

            def redraw_playback_text(current_token_index: int):
                """Clears and redraws all text up to the specified token index."""
                playback_text.plain = ""
                for i in range(current_token_index):
                    token_obj = tokens[i]
                    clean_token = token_obj.token.strip()
                    if clean_token:
                        colorized = Text.from_ansi(colorize_token(clean_token, token_obj.loudness))
                        playback_text.append(colorized)
                        if not (clean_token.endswith('.') or clean_token.endswith('!') or clean_token.endswith('?')):
                            playback_text.append(" ", style="white on black")
                live.refresh()


            while not playstate.quit:
                # --- Handle Keyboard Input ---
                if select.select([sys.stdin], [], [], 1.0 / FPS) == ([sys.stdin], [], []):
                    keys = ''
                    # Read all available input characters without blocking
                    while select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                        keys += sys.stdin.read(1)
                    
                    if keys:
                        logging.info(f"Key received: {keys!r}")
                        update_logs()

                    if keys == 'p' or keys == ' ':
                        playstate.playing = not playstate.playing
                        playstate.toggle_pause_request = True
                    elif keys == 'r' and not is_loaded_run:
                        playstate.restart = True
                    elif keys == 'h':
                        playstate.seek_to_start = True
                    elif keys == 'q':
                        playstate.quit = True
                    elif keys == '\x1b[D':  # Left arrow
                        playstate.seek_by = -20
                    elif keys == '\x1b[C':  # Right arrow
                        playstate.seek_by = 20
                
                if playstate.restart:
                    playback.stop()
                    # A bit of a hack to break out of the 'with Live' context
                    # by triggering an exception that we catch immediately.
                    raise Exception("Restarting")

                if playstate.toggle_pause_request:
                    if playback.paused:
                        playback.resume()
                    else:
                        playback.pause()
                    playstate.toggle_pause_request = False

                # --- Handle Seeking (can happen whether paused or playing) ---
                if playstate.seek_by != 0:
                    inew = max(0, min(inext + playstate.seek_by, len(tokens) - 1))
                    if inew != inext:
                        inext = inew
                        t = tokens[inext].timestamp
                        playback.seek(t)
                        redraw_playback_text(inext)
                    playstate.seek_by = 0

                if playstate.seek_to_start:
                    inext = 0
                    t = 0.0
                    playback.seek(t)
                    redraw_playback_text(inext)
                    playstate.seek_to_start = False

                # --- Handle Input & State ---
                if playstate.playing:
                    pbar.update(pbar_task, description="Playing")

                    # We need catchup_rate to be defined for the UI text, initialize it here.
                    catchup_rate = 0.0 

                    if playstate.seek_by != 0:
                        # This logic is now handled above, outside the playing/paused branch
                        pass
                    else:
                        # --- Time & Loudness Update (only when not seeking) ---
                        t_playback = playback.curr_pos
                        f = min(int(t_playback * config.tps), lufs.shape[0] - 1)
                        
                        K_MIN = 0.05
                        K_MAX = 8.0
                        dt = 1.0 / FPS
                        lag = t_playback - t
                        lag_t = float(clamp01(lag / 0.5))
                        
                        catchup_rate = K_MIN + (K_MAX - K_MIN) * sc(v_lufs[f], lag_t * 0.5)
                        
                        t += dt * catchup_rate

                        # This clamp, while intended to keep virtual time from getting ahead
                        # of audio time, breaks seeking. The audio player doesn't update
                        # its position instantly, so this clamp immediately cancels out the
                        # virtual time change from a seek.
                        t = float(maths.clamp(t, 0, t_playback))

                    # --- Loop / End Condition ---
                    if inext >= len(tokens):
                        inext = 0
                        t = 0.0
                        playback_text.plain = ""
                        # Restart playback from the beginning
                        playback.play()
                        live.refresh()
                        continue
                    
                    # --- UI Update ---
                    # We still need the current playback time for the UI, even if we just sought
                    t_playback = playback.curr_pos
                    f = min(int(t_playback * config.tps), lufs.shape[0] - 1)
                    
                    dynamic_info_text.plain = (
                        f"Audio Time: {t_playback:.2f}s / {song_duration:.2f}s\n"
                        f"Virtual Time: {t:.2f}s\n"
                        f"Tokens: {inext} / {len(tokens)}\n"
                        f"Loudness: {lufs[f]:.3f}\n"
                        f"Catch-up rate: {catchup_rate:.2f}x"
                    )
                    pbar.update(pbar_task, completed=inext)
                    
                    newly_displayed_text = False

                    while inext < len(tokens) and tokens[inext].timestamp <= t:
                        token_obj = tokens[inext]
                        token = token_obj.token
                        
                        clean_token = token.strip()
                        if clean_token:
                            colorized = Text.from_ansi(colorize_token(clean_token, token_obj.loudness))
                            playback_text.append(colorized)
                            if not (clean_token.endswith('.') or clean_token.endswith('!') or clean_token.endswith('?')):
                                 playback_text.append(" ", style="white on black")
                        
                        inext += 1
                        newly_displayed_text = True
                    
                    if newly_displayed_text:
                        live.refresh()

                else: # Paused
                    if playback.active:
                        playback.pause()

                    # Update UI
                    dynamic_info_text.plain = "[bold red]PAUSED[/]\n\n" + dynamic_info_text.plain.split('\n\n')[-1]
                    pbar.update(pbar_task, description="Paused")
                    live.refresh()

            # --- Cleanup ---
            if playback.active:
                playback.stop()
            
            header_text.plain = "Playback finished."
            dynamic_info_text.plain = "Done."
            live.refresh()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    # After the live display is finished, print the final output
    final_console.print(Panel(Text(config.prompt, justify="center"), title="Prompt", border_style="green"))
    final_text = Text()
    for token_obj in tokens:
        final_text.append(token_obj.token)
    final_console.print(Panel(final_text, title="Final Generated Text", border_style="blue"))
    
    return tokens


def main(config: AppConfig):
    """
    Main application logic: processes audio, shows plot, and runs the TUI.
    """
    console = Console()
    
    # Logging setup
    log_stream = io.StringIO()
    log_console = Console(file=log_stream, force_terminal=True, width=120)
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        datefmt="[%X]",
        handlers=[RichHandler(console=log_console, rich_tracebacks=True, show_path=False)],
        force=True
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    # BAKE
    # ----------------------------------------
    logging.info("Loading and processing audio data...")
    maths.fps = config.tps
    lufs_raw = audio.load_lufs(config.song_path, fps=config.tps)
    maths.n = lufs_raw.shape[0]

    if lufs_raw.shape[0] == 0:
        logging.error("Could not load loudness data or song is silent.")
        console.print(Text.from_ansi(log_stream.getvalue()))
        return
    
    lufs = maths.scurve(maths.wavg(lufs_raw, window=0.1), 0.75, 3.0)
    v_lufs = maths.absdiff(lufs)
    if np.max(v_lufs) > 0:
        v_lufs = v_lufs / np.max(v_lufs)
    else:
        v_lufs = np.zeros_like(v_lufs)
    temp = config.min_temp + (config.max_temp - config.min_temp) * v_lufs
    logging.info("Audio data processed.")
    logging.info("Displaying loudness analysis graph. Close the window to continue.")
    
    console.print(Panel(Text.from_ansi(log_stream.getvalue()), title="[bold cyan]Startup[/]", border_style="cyan"))

    # Matplotlib Plot
    # ----------------------------------------
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    frame_axis = np.arange(lufs_raw.shape[0])

    ax1.plot(frame_axis, lufs_raw)
    ax1.set_title("Original Loudness")
    ax1.set_ylabel("Normalized Loudness (0-1)")

    ax2.plot(frame_axis, lufs)
    ax2.set_title("Processed Loudness (Smoothed + S-Curve)")
    ax2.set_ylabel("Processed Loudness (0-1)")

    ax3.plot(frame_axis, v_lufs)
    ax3.set_title("Loudness Velocity")
    ax3.set_ylabel("Velocity (0-1)")

    ax4.plot(frame_axis, temp)
    ax4.set_title("Effective Temperature")
    ax4.set_xlabel("Time (tokens/frames)")
    ax4.set_ylabel("Temperature")

    plt.tight_layout()
    plt.show()

    # --- TUI Execution ---
    timed_tokens = run_tui(config, temp, lufs, lufs_raw, v_lufs)

    if config.save_run_path and timed_tokens:
        logging.info(f"Saving run to {config.save_run_path}...")
        run_data = GenerationRun(
            config=config,
            timed_tokens=timed_tokens,
            temp=temp,
            lufs=lufs,
            lufs_raw=lufs_raw,
            v_lufs=v_lufs
        )
        run_data.save(config.save_run_path)
        logging.info("Run saved.")


def app_main():
    while True:
        try:
            parser = argparse.ArgumentParser(description="Audio-reactive token generation with LLMs.")
            parser.add_argument("--song-path", type=Path, help="Path to the song file.", default="inputs/miles-pharaoh.flac")
            parser.add_argument("--prompt", help="Prompt for the LLM.", default="Describe a beautiful sunset in ~600 words.")
            parser.add_argument("--max-tokens", type=int, default=50, help="Maximum number of tokens to generate.")
            
            # Action-specific arguments
            parser.add_argument("--load-run", type=Path, help="Load a saved run from a file and start playback.")
            parser.add_argument("--save-run", type=Path, help="Override default save path. By default, all runs are saved to .runs/")

            args = parser.parse_args()

            if args.load_run:
                try:
                    run_data = GenerationRun.load(args.load_run)
                    Console().print(f"Successfully loaded run from [bold cyan]{args.load_run}[/bold cyan]")
                    run_tui(
                        config=run_data.config,
                        temp=run_data.temp,
                        lufs=run_data.lufs,
                        lufs_raw=run_data.lufs_raw,
                        v_lufs=run_data.v_lufs,
                        timed_tokens=run_data.timed_tokens,
                        is_loaded_run=True
                    )
                except FileNotFoundError:
                    Console().print(f"[bold red]Error: Run file not found at {args.load_run}[/bold red]")
                except Exception as e:
                    Console().print(f"[bold red]An error occurred while loading the run: {e}[/bold red]")
                
                break # Exit after loading a run

            config = AppConfig(
                song_path=args.song_path, 
                prompt=args.prompt, 
                max_tokens=args.max_tokens,
                save_run_path=args.save_run # Pass user-provided path, or None
            )
            main(config)
        except Exception as e:
            if "Restarting" in str(e):
                continue
            else:
                Console().print_exception()
                break
        break


if __name__ == "__main__":
    app_main()
# 