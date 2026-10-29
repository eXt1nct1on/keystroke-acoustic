"""Minimal Streamlit interface for unlabeled keystroke inference."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from audio_utils import detect_onsets
from infer import class_names_from_directory, infer, key_to_text


MODEL_PATH = PROJECT_ROOT / "models" / "keystroke_classifier.keras"
CLASSES_DIR = PROJECT_ROOT / "data" / "spectrograms" / "train"


@st.cache_resource
def load_model() -> tf.keras.Model:
    return tf.keras.models.load_model(MODEL_PATH)


def load_audio(uploaded_file: object) -> tuple[np.ndarray, int]:
    data = uploaded_file.getvalue()
    audio, sample_rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return audio.mean(axis=1), sample_rate


def plot_results(
    audio: np.ndarray,
    sample_rate: int,
    onset_times: np.ndarray,
    predictions: list[tuple[str, float]],
) -> plt.Figure:
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(12, 6),
        gridspec_kw={"height_ratios": [2, 1]},
    )
    time_axis = np.arange(len(audio)) / sample_rate
    axes[0].plot(time_axis, audio, linewidth=0.6)
    for onset_time in onset_times:
        axes[0].axvline(onset_time, color="tab:red", alpha=0.6, linewidth=1)
    axes[0].set_title("Detected keystroke onsets")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Amplitude")

    labels = [
        key_to_text(key).replace("\n", "return") or "space"
        for key, _ in predictions
    ]
    confidences = [confidence for _, confidence in predictions]
    axes[1].bar(range(len(predictions)), confidences, color="tab:blue")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Confidence")
    axes[1].set_xlabel("Detected keystroke")
    axes[1].set_xticks(range(len(labels)), labels)
    axes[1].tick_params(axis="x", rotation=45)
    figure.tight_layout()
    return figure


def main() -> None:
    st.title("Keystroke Acoustic Inference")

    if hasattr(st, "audio_input"):
        audio_file = st.audio_input("Record or upload a WAV file")
    else:
        audio_file = st.file_uploader("Upload a WAV file", type=["wav"])

    if not st.button("Run", type="primary"):
        return
    if audio_file is None:
        st.warning("Provide a WAV recording before running inference.")
        return

    try:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        class_names = class_names_from_directory(CLASSES_DIR)
        model = load_model()
        audio, sample_rate = load_audio(audio_file)
        onset_times = detect_onsets(audio, sample_rate)
        predictions = infer(audio, sample_rate, model, class_names, offset=0.0)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        st.error(f"Inference failed: {error}")
        return

    raw_text = "".join(key_to_text(key) for key, _ in predictions)
    st.subheader("Predicted text")
    st.code(raw_text or "(no keystrokes detected)")

    if predictions:
        figure = plot_results(audio, sample_rate, onset_times, predictions)
        st.pyplot(figure)
        plt.close(figure)
    else:
        st.info("No keystroke onsets were detected.")


if __name__ == "__main__":
    main()