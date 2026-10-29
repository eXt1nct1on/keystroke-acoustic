"""Infer keystrokes from an unlabeled audio recording."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from spellchecker import SpellChecker
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    from audio_utils import detect_onsets, extract_clip, load_audio, spectrogram_image
except ImportError:
    detect_onsets = None
    extract_clip = None
    load_audio = None
    spectrogram_image = None


SAMPLE_RATE = 44_100
DEFAULT_DURATION = 5.0
DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/keystroke-acoustic"
DEFAULT_SPECTROGRAM_DIR = "data/spectrograms"
DEFAULT_MODEL = "models/keystroke_classifier.keras"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict keystrokes from a recorded or supplied audio file."
    )
    parser.add_argument("--audio", type=Path, help="Audio file to analyze.")
    parser.add_argument(
        "--record-seconds",
        type=float,
        default=DEFAULT_DURATION,
        help="Seconds to record when --audio is omitted (default: 5).",
    )
    parser.add_argument(
        "--record-output",
        type=Path,
        help="Optional path for saving a newly recorded WAV file.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "models"
        / "keystroke_classifier.keras",
        help="Trained Keras model path.",
    )
    parser.add_argument(
        "--classes-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "data"
        / "spectrograms"
        / "train",
        help="Directory whose subfolder names define model class order.",
    )
    parser.add_argument(
        "--offset",
        type=float,
        default=0.0,
        help="Seconds to shift each detected onset before slicing.",
    )
    parser.add_argument(
        "--autocorrect",
        action="store_true",
        help="Apply word-level spell correction to the predicted text.",
    )
    parser.add_argument(
        "--drive-root",
        default=os.environ.get("KEYSTROKE_DRIVE_ROOT", DEFAULT_DRIVE_ROOT),
        help="Reserved for Colab workflows; paths may be supplied directly.",
    )
    return parser.parse_args()


def record_audio(duration: float, sample_rate: int) -> tuple[np.ndarray, int]:
    if sd is None:
        raise RuntimeError(
            "Recording requires sounddevice. Install with "
            "'python -m pip install sounddevice'."
        )
    if duration <= 0:
        raise ValueError("Recording duration must be positive.")
    print(f"Recording for {duration:.1f} seconds...")
    audio = sd.rec(
        int(round(duration * sample_rate)),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    return np.asarray(audio[:, 0]), sample_rate


def load_or_record(args: argparse.Namespace) -> tuple[np.ndarray, int]:
    if args.audio is not None:
        if load_audio is None:
            raise RuntimeError(
                "Loading requires librosa. Install project requirements first."
            )
        return load_audio(args.audio)

    audio, sample_rate = record_audio(args.record_seconds, SAMPLE_RATE)
    if args.record_output is not None:
        if sf is None:
            raise RuntimeError(
                "Saving recordings requires soundfile. Install project requirements first."
            )
        args.record_output.parent.mkdir(parents=True, exist_ok=True)
        sf.write(args.record_output, audio, sample_rate)
        print(f"Recording saved to {args.record_output}")
    return audio, sample_rate


def class_names_from_directory(classes_dir: Path) -> list[str]:
    if not classes_dir.is_dir():
        raise FileNotFoundError(f"Class directory does not exist: {classes_dir}")
    class_names = sorted(path.name for path in classes_dir.iterdir() if path.is_dir())
    if not class_names:
        raise ValueError(f"No class folders found in {classes_dir}")
    return class_names


def key_to_text(key: str) -> str:
    normalized = key.strip().lower()
    if normalized in {"space", "sp"}:
        return " "
    if normalized in {"enter", "return", "newline"}:
        return "\n"
    return key


def autocorrect_text(raw_text: str) -> str:
    spellchecker = SpellChecker()
    corrected_words: list[str] = []
    for line in raw_text.splitlines(keepends=True):
        ending = "\n" if line.endswith("\n") else ""
        words = line.rstrip("\n").split(" ")
        corrected_words.append(
            " ".join(spellchecker.correction(word) or word for word in words)
            + ending
        )
    return "".join(corrected_words)


def infer(
    audio: np.ndarray,
    sample_rate: int,
    model: tf.keras.Model,
    class_names: list[str],
    offset: float,
) -> list[tuple[str, float]]:
    if detect_onsets is None or extract_clip is None or spectrogram_image is None:
        raise RuntimeError(
            "Inference requires librosa, matplotlib, and pillow. "
            "Install project requirements first."
        )
    onset_times = detect_onsets(audio, sample_rate)
    predictions: list[tuple[str, float]] = []
    for onset_time in onset_times:
        clip = extract_clip(
            audio,
            sample_rate,
            float(onset_time),
            offset=offset,
            refine_onset=False,
        )
        image = np.asarray(spectrogram_image(clip, sample_rate), dtype=np.float32)
        probabilities = model.predict(
            preprocess_input(image)[np.newaxis, ...], verbose=0
        )[0]
        class_index = int(np.argmax(probabilities))
        predictions.append((class_names[class_index], float(probabilities[class_index])))
    return predictions


def main() -> int:
    args = parse_args()
    try:
        if not args.model.is_file():
            raise FileNotFoundError(f"Trained model does not exist: {args.model}")
        class_names = class_names_from_directory(args.classes_dir)
        model = tf.keras.models.load_model(args.model)
        audio, sample_rate = load_or_record(args)
        predictions = infer(audio, sample_rate, model, class_names, args.offset)
        raw_text = "".join(key_to_text(key) for key, _ in predictions)

        print(f"\nRaw prediction: {raw_text!r}")
        print("Per-character confidence:")
        for index, (key, confidence) in enumerate(predictions, start=1):
            print(f"  {index:>3}: {key!r} ({confidence:.2%})")
        print(f"Detected keystrokes: {len(predictions)}")
        if args.autocorrect:
            print(f"Autocorrected: {autocorrect_text(raw_text)!r}")
    except (
        FileNotFoundError,
        ImportError,
        RuntimeError,
        ValueError,
        OSError,
    ) as error:
        print(f"Inference failed: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())