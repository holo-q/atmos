# Atmos

*Audio-Reactive Sampling in LLMs*

This project is a proof-of-concept for exploring a novel method of influencing Large Language Model (LLM) inference by dynamically controlling the sampling temperature with the loudness of a music track.

## The Hypothesis

An hypothesis suggests that human music encodes the fundamental stochastic pattern generator of the human brain. By tying this stochasticity to the temperature parameter, the model's latent space is explored with a human rhythm.

In effect, a music's loudness acts like a pre-sampled probability distribution of a human brain's softmax of raw universal pattern recognition and prediction. We can tune the LLM's sampling to this same rendered world frequency.

For now, this project allows for comparing the rhythmic effect of various styles of music on the text-generation of LLMs when tied to temperature, including jazz, techno, and silence as a control.

## How It Works

The script processes an audio file to create a loudness contour, which is then used to modulate the temperature parameter during token-by-token generation from an LLM.

1.  **Audio Analysis**: The input audio is loaded and its loudness (LUFS) is analyzed on a frame-by-frame basis, matching the desired token-per-second rate.
2.  **Velocity Calculation**: This raw loudness data is smoothed, and its rate of change (velocity) is calculated. This "loudness velocity" captures the dynamic intensity of the music.
3.  **Temperature Mapping**: The velocity is mapped to a temperature range. Quiet, stable parts of the song result in a low, conservative temperature, while loud, dynamic peaks result in a high, more creative temperature.
4.  **Reactive Generation**: The script connects to a local LLM server. For each token, it sends the history and the calculated temperature for that specific moment in the song.
5.  **Interactive Playback**: A Text-based User Interface (TUI) displays the generated text in real-time, synchronized with the music playback. You can pause, seek, and restart the generation.
6.  **Saving and Loading**: Each generation run, including the prompt, audio analysis data, and generated tokens, can be saved and replayed later.

## Setup

1.  **Install Python dependencies:**

    ```bash
    pip install -r requirements.txt
    ```
    *(Note: If you have issues, you might need to install PyAudio manually or via your system's package manager, as it has non-Python dependencies.)*

2.  **Install system dependencies for audio playback (if needed):**

    On Debian/Ubuntu:
    ```bash
    sudo apt-get install libportaudio2
    ```

    On Fedora:
    ```bash
    sudo dnf install portaudio
    ```

    On Arch Linux:
    ```bash
    sudo pacman -S portaudio
    ```

3.  **Run a local LLM server:**

    This project is designed to work with a local LLM server that exposes an OpenAI-compatible API. We recommend using [LM Studio](https://lmstudio.ai/).

    -   Download and install LM Studio.
    -   From the search tab, download a model.
    -   Start the server on `localhost:1234`.

## Usage

### Generate a new response

Run the script from the command line. You can run it without arguments to use the default song and prompt:
```bash
python main.py
```

Or, you can provide your own song and prompt:

```bash
python main.py --song-path "path/to/your/song.mp3" --prompt "Why is the sky blue?"
```

The song will start playing and the TUI will launch, showing the text generation in real-time.

### Load a previous run

You can load and replay a saved run from a `.npz` file:
```bash
python main.py --load-run .runs/run_YYYY-MM-DD_HH-MM-SS.npz
```

## The Vision: Where We Go From Here

This is only a proof of concept. The possibilities are endless—these things are like lego blocks. You can route energy around and let attractor dynamics sort out the model. LLMs are dynamical and physical systems on a world model.

If the hypothesis is right, then the real tour de force is in using the Magenta RT music generation model as the model's stochasticity generator. We use RL/GRPO to reinforce a side-chained context in which the model is writing music generator prompts parallel to the main conversation context.

From then on, LLM inference is augmented by a side-reasoning stochasticity prompter which is keying the music generation over the course of text generation in the main conversation context. This creates strange attractor dynamics in which the model is learning to take itself out of a basin with music.

A more direct approach is being researched (and already applied) at frontier labs where instead the model performs continuous-space reasoning which is gated into a learnt linear map producing the temperature, which they can do reinforcement learning on as well. Instead of the side-chained context, they train the model with the ability to reason about a desired temperature at each token, a micro-reasoning which is spliced in before sampling.

Essentially they're trying to use the LLM itself to generate the temperature. This is inferior since it has to climb gradients in reverse rather than transplanting morphisms. (but probably cheaper and produces results more readily)

We are only getting warmed up.
