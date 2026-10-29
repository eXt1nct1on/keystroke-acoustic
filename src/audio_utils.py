"""Shared audio slicing and mel-spectrogram utilities."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


CLIP_DURATION_SECONDS = 0.25
N_MELS = 128
IMAGE_SIZE = 224
IMAGE_DPI = 100


def load_audio(audio_path: Path) -> tuple[np.ndarray, int]:
    """Load an audio file as mono without resampling."""
    audio, sample_rate = librosa.load(audio_path, sr=None, mono=True)
    return audio.astype(np.float32, copy=False), sample_rate


def detect_onsets(
    audio: np.ndarray,
    sample_rate: int,
    *,
    backtrack: bool = False,
) -> np.ndarray:
    """Return acoustic onset times in seconds."""
    return librosa.onset.onset_detect(
        y=audio,
        sr=sample_rate,
        units="time",
        backtrack=backtrack,
    )


def extract_samples(audio: np.ndarray, start: int, length: int) -> np.ndarray:
    """Return exactly length samples, zero-padding outside the recording."""
    clip = np.zeros(length, dtype=np.float32)
    source_start = max(start, 0)
    source_end = min(start + length, len(audio))
    if source_end > source_start:
        destination_start = source_start - start
        clip[destination_start : destination_start + source_end - source_start] = (
            audio[source_start:source_end]
        )
    return clip


def extract_clip(
    audio: np.ndarray,
    sample_rate: int,
    timestamp: float,
    offset: float = 0.0,
    refine_onset: bool = False,
) -> np.ndarray:
    """Extract a 250 ms clip centered on an event, optionally refined to onset."""
    clip_samples = round(CLIP_DURATION_SECONDS * sample_rate)
    initial_center = timestamp + offset
    initial_start = round(
        (initial_center - CLIP_DURATION_SECONDS / 2) * sample_rate
    )
    initial_clip = extract_samples(audio, initial_start, clip_samples)

    if refine_onset:
        onset_frames = librosa.onset.onset_detect(
            y=initial_clip,
            sr=sample_rate,
            units="samples",
            backtrack=False,
        )
        if len(onset_frames) > 0:
            onset_center = initial_start + int(onset_frames[0])
            return extract_samples(
                audio, onset_center - clip_samples // 2, clip_samples
            )
    return initial_clip


def mel_spectrogram(clip: np.ndarray, sample_rate: int) -> np.ndarray:
    """Convert a clip to a decibel mel-spectrogram."""
    mel = librosa.feature.melspectrogram(y=clip, sr=sample_rate, n_mels=N_MELS)
    return librosa.power_to_db(mel, ref=np.max)


def spectrogram_image(clip: np.ndarray, sample_rate: int) -> Image.Image:
    """Render a spectrogram exactly as a 224x224 RGB image."""
    figure = plt.figure(figsize=(IMAGE_SIZE / IMAGE_DPI, IMAGE_SIZE / IMAGE_DPI))
    axes = figure.add_axes((0, 0, 1, 1))
    axes.imshow(
        mel_spectrogram(clip, sample_rate),
        origin="lower",
        aspect="auto",
        cmap="magma",
    )
    axes.axis("off")
    image_buffer = BytesIO()
    figure.savefig(
        image_buffer,
        dpi=IMAGE_DPI,
        format="png",
        bbox_inches=None,
        pad_inches=0,
    )
    plt.close(figure)
    image_buffer.seek(0)
    with Image.open(image_buffer) as image:
        return image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))


def save_spectrogram(
    clip: np.ndarray, sample_rate: int, output_path: Path
) -> None:
    """Render and save one fixed-size spectrogram PNG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    spectrogram_image(clip, sample_rate).save(output_path, format="PNG")
