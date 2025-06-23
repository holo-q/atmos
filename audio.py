"""
Common audio processing functionalities using librosa and more.
There are caching mechanisms to avoid recalculating the same thing over and over again.
All audio data is normalized to [0, 1]

# Important note about caching

1. When the audio is in a hidden folder, the cache is saved in the parent folder.
   This is because music files are huge and we want to skip sending them to deploy servers (VastAI)
   Instead, we will only send the cache files, which are much smaller, and so we need to save them in a non-hidden folder.

2. When the audio file is hidden, the cached data is always unhidden.
   This is because the cached data is small and likely to be used.

"""

import logging
from pathlib import Path

import soundfile
import numpy as np
import librosa
import resampy


log = logging.getLogger("audio")


def norm(data):
    """Normalize data to [0, 1] range."""
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val - min_val > 0:
        return (data - min_val) / (max_val - min_val)
    return np.zeros_like(data)


def zero():
    return np.array([])


def load_audio_cache(filepath, cachename, fps):
    filepath = Path(filepath)
    cachepath = get_audio_cachepath(filepath, cachename, fps)
    if cachepath.exists():
        # print(f"audio.{cachename}({filepath.stem}): loading cache")
        return np.load(cachepath.as_posix())

    raise Exception(f"Cache file {cachepath} does not exist")


def get_audio_cachepath(filepath, cachename, fps):
    filepath = Path(filepath)
    # If the file is in a hidden folder, we save the cache in the parent folder
    if filepath.parent.stem.startswith("."):
        filepath = filepath.parent.parent / filepath.name

    # If the data would be hidden, we unhide. Cached audio data should never be hidden!
    if filepath.name.startswith("."):
        filepath = filepath.with_name(filepath.name[1:])

    cachepath = filepath.with_stem(f"{Path(filepath).stem}_{cachename}_{fps}").with_suffix(
        ".npy"
    )
    return cachepath


def save_audio_cache(filepath, cachename, arr, enable, fps):
    cachepath = get_audio_cachepath(filepath, cachename, fps)
    if enable:
        np.save(cachepath.as_posix(), arr)
    return arr


def has_audio_cache(filepath, cachename, enable, fps):
    cachepath = get_audio_cachepath(filepath, cachename, fps)
    if enable and not cachepath.exists():
        log.info(f"audio.{cachename}({filepath.stem}): missing cache, calculating ...")
    return enable and cachepath.exists()


def load_crepe_keyframes(filepath, fps=60):
    import pandas as pd

    df = pd.read_csv(filepath)
    freq = to_keyframes(df["frequency"], len(df["frequency"]) / df["time"].values[-1], fps)
    confidence = to_keyframes(
        df["confidence"], len(df["frequency"]) / df["time"].values[-1], fps
    )
    return freq, confidence


def load_rosa(filepath, fps=60):
    import librosa

    y, sr = soundfile.read(filepath)
    y = librosa.to_mono(y.T)  # Convert stereo to mono if required

    # Calculate the duration of the audio file
    print("load_rosa: get_duration")
    duration = librosa.get_duration(y=y, sr=sr)

    # Onset detection
    print("load_rosa: onset_strength")
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    original_fps = len(onset_env) / duration
    onset_env_resampled = resampy.resample(onset_env, original_fps, fps)

    # Tempo and beat detection
    print("load_rosa: beat_track")
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_changes = np.zeros_like(onset_env)
    beat_changes[beat_frames] = 1
    beat_changes_resampled = resampy.resample(beat_changes, original_fps, fps)

    # Harmonic and percussive source separation
    print("load_rosa: hpss")
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # Chroma and harmonic changes
    print("load_rosa: calculating chroma changes")
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)[1]
    chroma_resampled = resampy.resample(chroma, original_fps, fps)
    # chroma_changes = np.diff(chroma, axis=1)
    # chroma_changes = np.sum(np.abs(chroma_changes), axis=0)
    # chroma_changes = np.pad(chroma_changes, (0, len(onset_env) - len(chroma_changes)), mode='constant')
    # chroma_changes_resampled = resampy.resample(chroma_changes, original_fps, fps)
    # chroma_changes_resampled = np.where(chroma_changes_resampled > np.percentile(chroma_changes_resampled, 75), 1, 0)

    # Calculate the duration of the audio file
    duration = librosa.get_duration(y=y, sr=sr)

    # Spectral analysis
    print("load_rosa: calculating spectral changes")
    S = np.abs(librosa.stft(y))
    spectral_contrast = librosa.feature.spectral_contrast(S=S, sr=sr)[1]
    spectral_contrast_resampled = resampy.resample(spectral_contrast, original_fps, fps)
    # spectral_changes = np.diff(spectral_contrast, axis=1)
    # spectral_changes = np.sum(np.abs(spectral_changes), axis=0)
    # spectral_changes = np.pad(spectral_changes, (0, len(S[0]) - len(spectral_changes)), mode='constant')
    # original_fps = len(S[0]) / duration
    # spectral_changes_resampled = resampy.resample(spectral_changes, original_fps, fps)
    # spectral_changes_resampled = np.where(spectral_changes_resampled > np.percentile(spectral_changes_resampled, 75), 1, 0)

    # Timbre analysis (MFCC)
    print("load_rosa: calculating timbre changes")
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    mfcc_resampled = resampy.resample(mfcc[1], original_fps, fps)
    # mfcc_changes = np.diff(mfcc, axis=1)
    # mfcc_changes = np.sum(np.abs(mfcc_changes), axis=0)
    # mfcc_changes = np.pad(mfcc_changes, (0, len(S[0]) - len(mfcc_changes)), mode='constant')
    # mfcc_changes_resampled = resampy.resample(mfcc_changes, original_fps, fps)
    # mfcc_changes_resampled = np.where(mfcc_changes_resampled > np.percentile(mfcc_changes_resampled, 75), 1, 0)

    # Combine changes from different techniques
    # combined_changes = onset_times + beat_changes_resampled + chroma_changes_resampled
    # combined_changes = np.where(combined_changes > 0, 1, 0)

    # Spectral bandwidth
    freqs, times, D = librosa.reassigned_spectrogram(y, fill_nan=True)
    bandwidth = librosa.feature.spectral_bandwidth(S=np.abs(D), freq=freqs)
    bandwidth_resampled = resampy.resample(bandwidth[0], original_fps, fps)

    # Spectral flatness
    flatness = librosa.feature.spectral_flatness(y=y)
    flatness_resampled = resampy.resample(flatness[0], original_fps, fps)

    # Spectral bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    bandwidth_resampled = resampy.resample(bandwidth[0], original_fps, fps)

    # Sentiment
    sentiment = happy_sad(filepath)

    # return zero(), onset_env_resampled, beat_changes_resampled, chroma_changes_resampled, spectral_changes_resampled, mfcc_resampled
    return (
        zero(),
        onset_env_resampled,
        beat_changes_resampled,
        chroma_resampled,
        spectral_contrast_resampled,
        mfcc_resampled,
        flatness_resampled,
        bandwidth_resampled,
        sentiment,
    )


def load_lufs(filepath, caching=True, fps=60):
    import soundfile as sf
    from loudness import lufs_meter

    if not has_audio_cache(filepath, "lufs", caching, fps):
        y, sr = sf.read(filepath)  # load audio (with shape (samples, channels))
        # y = filter.butter(y, sr, 'highpass', 1, 400)
        meter = lufs_meter(sr, 1 / 60, overlap=0)
        loudness = meter.get_mlufs(y)

        # Replace infinities and nans with zero
        loudness[np.isinf(loudness)] = 0

        # WARNING: WORKAROUND AHEAD
        # the loudness values returned are 0 when the audio is silent
        # so we get all values above a threshold and set them to the meter's minimum
        silence_threshold = -5
        loudness = np.where(loudness > silence_threshold, meter.threshold, loudness)

        # Normalize to 0-1 in a 12 second window
        loudness = norm(loudness)
        loudness = resampy.resample(loudness, 60, fps)

        return save_audio_cache(filepath, "lufs", loudness, caching, fps)

    return load_audio_cache(filepath, "lufs", fps)


def load_pca(filepath, num_components=3, caching=True, fps=60):
    import librosa
    from sklearn.decomposition import PCA

    if not has_audio_cache(filepath, "pca", caching, fps):
        y, sr = librosa.load(filepath)
        chromagram = librosa.feature.chroma_stft(
            y=y,
            sr=sr,
            hop_length=int(sr / fps),
            win_length=int(sr * 0.03),
            n_chroma=12,
        )
        pca = PCA(n_components=num_components)
        chromagram_pca = pca.fit_transform(
            chromagram.T
        )  # Transpose chromagram for PCA along the time axis
        chromagram_pca = chromagram_pca.T  # Transpose back to original shape

        # resample each row
        duration = librosa.get_duration(y=y, sr=sr)
        original_fps = len(chromagram_pca[0]) / duration

        ret = []
        for i in range(num_components):
            comp = resampy.resample(chromagram_pca[i], original_fps, fps)
            ret.append(comp)

        save_audio_cache(filepath, "pca", np.array(ret), caching, fps)
        return tuple(ret)

    arr = load_audio_cache(filepath, "pca", fps)
    ret = []
    for i in range(num_components):
        ret.append(arr[i])
    return tuple(ret)


def load_flatness(filepath, caching=True, fps=60):
    import librosa

    if not has_audio_cache(filepath, "flatness", caching, fps):
        y, sr = librosa.load(filepath)
        # Spectral flatness
        flatness = librosa.feature.spectral_flatness(
            y=y, hop_length=int(sr / fps), win_length=int(sr * 0.03)
        )[0]

        duration = librosa.get_duration(y=y, sr=sr)
        original_fps = len(flatness) / duration

        flatness_resampled = resampy.resample(flatness, original_fps, fps)
        return save_audio_cache(filepath, "flatness", flatness_resampled, caching, fps)

    return load_audio_cache(filepath, "flatness", fps)


def load_onset(filepath, caching=True, fps=60):
    import librosa

    if not has_audio_cache(filepath, "onset", caching, fps):
        y, sr = librosa.load(filepath)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_env = norm(onset_env)

        duration = librosa.get_duration(y=y, sr=sr)
        original_fps = len(onset_env) / duration
        onset_env_resampled = resampy.resample(onset_env, original_fps, fps)
        return save_audio_cache(filepath, "onset", onset_env_resampled, caching, fps)

    return load_audio_cache(filepath, "onset", fps)


def happy_sad(sound_file, fps=60):
    y, sr = soundfile.read(sound_file)
    y = librosa.to_mono(y.T)  # Convert stereo to mono if required

    # Extract chroma feature
    chroma = librosa.feature.chroma_cens(y=y, sr=sr)

    # Calculate major and minor chord templates
    major_template = chroma[[0, 4, 7], :]
    minor_template = chroma[[0, 3, 7], :]

    # Calculate happiness/sadness scores for each moment in time
    scores = np.sum(major_template, axis=0) - np.sum(minor_template, axis=0)
    scores = scores / (np.sum(major_template, axis=0) + np.sum(minor_template, axis=0))

    # Calculate the duration of the audio file
    duration = librosa.get_duration(y=y, sr=sr)
    original_fps = len(scores) / duration

    resampled_scores = resampy.resample(scores, original_fps, fps)

    return resampled_scores


def to_keyframes(dbs, original_sps, fps=60):
    start = 0
    total_seconds = len(dbs) / original_sps
    # print(len(dbs), original_sps, total_seconds)
    # start=0
    # total_seconds=5

    frames = int(fps * total_seconds)

    dt = np.zeros(frames)
    for i in range(frames):
        # frame --> seconds
        t = (i) / fps + start
        t1 = (i + 1) / fps + start
        # print(t, t1)

        d = dbs[int(t * original_sps) : int((t1) * original_sps)]
        dt[i] = np.mean(d)

        # remove infinities and nans
        if np.isinf(dt[i]) or np.isnan(dt[i]):
            dt[i] = dt[i - 1]

    return dt
    # return smooth_1euro(dt)
