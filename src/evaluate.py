"""Evaluate a trained keystroke classifier on the validation dataset."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/keystroke-acoustic"
DEFAULT_SPECTROGRAM_DIR = "data/spectrograms"
DEFAULT_MODEL = "models/keystroke_classifier.keras"
DEFAULT_CONFUSION_MATRIX = "models/confusion_matrix.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the keystroke classifier on validation spectrograms."
    )
    parser.add_argument(
        "--drive-root",
        default=os.environ.get("KEYSTROKE_DRIVE_ROOT", DEFAULT_DRIVE_ROOT),
        help="Google Drive project root.",
    )
    parser.add_argument(
        "--spectrogram-dir",
        default=None,
        help="Train/val directory, or relative path under --drive-root.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Trained model path, or relative path under --drive-root.",
    )
    parser.add_argument(
        "--confusion-matrix",
        default=None,
        help="Output image path, or relative path under --drive-root.",
    )
    return parser.parse_known_args()[0]


def mount_google_drive() -> None:
    try:
        from google.colab import drive
    except ImportError as error:
        raise RuntimeError(
            "This script is intended for Google Colab. "
            "Run it in Colab so Google Drive can be mounted."
        ) from error
    drive.mount("/content/drive")


def resolve_path(path: str, drive_root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else drive_root / candidate


def load_validation_dataset(
    spectrogram_dir: Path,
) -> tuple[tf.data.Dataset, list[str]]:
    train_dir = spectrogram_dir / "train"
    val_dir = spectrogram_dir / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError(
            f"Expected train and val directories under {spectrogram_dir}."
        )

    # Read classes from train first, then force validation to use that same order.
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )
    class_names = train_dataset.class_names
    if len(class_names) < 2:
        raise ValueError("At least two key-class folders are required.")

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        class_names=class_names,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )
    autotune = tf.data.AUTOTUNE
    validation_dataset = validation_dataset.map(
        lambda images, labels: (preprocess_input(tf.cast(images, tf.float32)), labels),
        num_parallel_calls=autotune,
    ).prefetch(autotune)
    return validation_dataset, class_names


def collect_predictions(
    model: tf.keras.Model, dataset: tf.data.Dataset
) -> tuple[np.ndarray, np.ndarray]:
    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for images, batch_labels in dataset:
        labels.append(batch_labels.numpy())
        predictions.append(np.argmax(model.predict(images, verbose=0), axis=1))
    if not labels:
        raise ValueError("The validation dataset contains no images.")
    return np.concatenate(labels), np.concatenate(predictions)


def save_confusion_matrix(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=np.arange(len(class_names)),
    )
    figure, axis = plt.subplots(figsize=(max(8, len(class_names) * 0.8), 8))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names,
    )
    display.plot(
        ax=axis,
        cmap="Blues",
        values_format="d",
        xticks_rotation=45,
        colorbar=False,
    )
    axis.set_title("Keystroke Validation Confusion Matrix")
    axis.set_xlabel("Predicted key")
    axis.set_ylabel("True key")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Confusion matrix saved to {output_path}")


def evaluate(
    model_path: Path,
    spectrogram_dir: Path,
    confusion_matrix_path: Path,
) -> None:
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model does not exist: {model_path}")

    validation_dataset, class_names = load_validation_dataset(spectrogram_dir)
    model = tf.keras.models.load_model(model_path)
    loss, accuracy = model.evaluate(validation_dataset, verbose=1)
    true_labels, predicted_labels = collect_predictions(model, validation_dataset)

    print(f"\nValidation loss: {loss:.4f}")
    print(f"Validation accuracy: {accuracy:.4f}")
    print("\nPer-class classification report:")
    print(
        classification_report(
            true_labels,
            predicted_labels,
            labels=np.arange(len(class_names)),
            target_names=class_names,
            zero_division=0,
        )
    )
    save_confusion_matrix(
        true_labels, predicted_labels, class_names, confusion_matrix_path
    )


def main() -> int:
    args = parse_args()
    try:
        mount_google_drive()
        drive_root = Path(args.drive_root)
        evaluate(
            resolve_path(args.model or DEFAULT_MODEL, drive_root),
            resolve_path(args.spectrogram_dir or DEFAULT_SPECTROGRAM_DIR, drive_root),
            resolve_path(
                args.confusion_matrix or DEFAULT_CONFUSION_MATRIX, drive_root
            ),
        )
    except (
        FileNotFoundError,
        ImportError,
        RuntimeError,
        ValueError,
        OSError,
    ) as error:
        print(f"Evaluation failed: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())