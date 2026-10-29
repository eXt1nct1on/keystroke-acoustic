"""Train a MobileNetV2 keystroke classifier in Google Colab."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 10
INITIAL_LEARNING_RATE = 1e-4
FINE_TUNE_LEARNING_RATE = 1e-5
DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/keystroke-acoustic"
DEFAULT_SPECTROGRAM_DIR = "data/spectrograms"
DEFAULT_CHECKPOINT = "models/keystroke_classifier_best.keras"
DEFAULT_FINAL_MODEL = "models/keystroke_classifier.keras"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the keystroke classifier using a Google Drive dataset."
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
        "--checkpoint",
        default=None,
        help="Best-model checkpoint path, or relative path under --drive-root.",
    )
    parser.add_argument(
        "--final-model",
        default=None,
        help="Final-model path, or relative path under --drive-root.",
    )
    # Colab may pass its own command-line arguments; ignore unknown ones.
    return parser.parse_known_args()[0]


def mount_google_drive() -> None:
    """Mount Drive when running in Colab and fail clearly elsewhere."""
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


def load_datasets(spectrogram_dir: Path) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    train_dir = spectrogram_dir / "train"
    val_dir = spectrogram_dir / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError(
            f"Expected train and val directories under {spectrogram_dir}."
        )

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=True,
        seed=42,
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
    train_dataset = train_dataset.map(
        lambda images, labels: (preprocess_input(tf.cast(images, tf.float32)), labels),
        num_parallel_calls=autotune,
    ).prefetch(autotune)
    validation_dataset = validation_dataset.map(
        lambda images, labels: (preprocess_input(tf.cast(images, tf.float32)), labels),
        num_parallel_calls=autotune,
    ).prefetch(autotune)
    return train_dataset, validation_dataset, class_names


def build_model(number_of_classes: int) -> tuple[tf.keras.Model, tf.keras.Model]:
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMAGE_SIZE, 3),
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(*IMAGE_SIZE, 3))
    features = base_model(inputs, training=False)
    features = layers.GlobalAveragePooling2D()(features)
    features = layers.Dropout(0.3)(features)
    outputs = layers.Dense(number_of_classes, activation="softmax")(features)
    model = tf.keras.Model(inputs, outputs, name="keystroke_mobilenetv2")
    return model, base_model


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def unfreeze_top_layers(base_model: tf.keras.Model, number_of_layers: int = 30) -> None:
    base_model.trainable = True
    cutoff = max(0, len(base_model.layers) - number_of_layers)
    for index, layer in enumerate(base_model.layers):
        # Batch normalization statistics remain stable during fine-tuning.
        layer.trainable = index >= cutoff and not isinstance(layer, layers.BatchNormalization)


def make_callbacks(checkpoint_path: Path) -> list[tf.keras.callbacks.Callback]:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
    ]


def combine_histories(
    initial_history: tf.keras.callbacks.History,
    fine_tune_history: tf.keras.callbacks.History,
) -> dict[str, list[float]]:
    keys = set(initial_history.history) | set(fine_tune_history.history)
    return {
        key: initial_history.history.get(key, [])
        + fine_tune_history.history.get(key, [])
        for key in keys
    }


def plot_training_curve(history: dict[str, list[float]]) -> None:
    epochs = range(1, len(history["loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["accuracy"], label="Train")
    axes[0].plot(epochs, history["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(epochs, history["loss"], label="Train")
    axes[1].plot(epochs, history["val_loss"], label="Validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    figure.tight_layout()
    plt.show()


def train(args: argparse.Namespace) -> Path:
    mount_google_drive()
    drive_root = Path(args.drive_root)
    spectrogram_dir = resolve_path(
        args.spectrogram_dir or DEFAULT_SPECTROGRAM_DIR, drive_root
    )
    checkpoint_path = resolve_path(
        args.checkpoint or DEFAULT_CHECKPOINT, drive_root
    )
    final_model_path = resolve_path(
        args.final_model or DEFAULT_FINAL_MODEL, drive_root
    )

    train_dataset, validation_dataset, class_names = load_datasets(spectrogram_dir)
    model, base_model = build_model(len(class_names))
    compile_model(model, INITIAL_LEARNING_RATE)
    callbacks = make_callbacks(checkpoint_path)

    print(f"Classes ({len(class_names)}): {class_names}")
    print("Training frozen MobileNetV2 base...")
    initial_history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=INITIAL_EPOCHS,
        callbacks=callbacks,
    )

    print("Fine-tuning the top 30 MobileNetV2 layers...")
    unfreeze_top_layers(base_model)
    compile_model(model, FINE_TUNE_LEARNING_RATE)
    fine_tune_history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        initial_epoch=INITIAL_EPOCHS,
        epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
        callbacks=callbacks,
    )

    final_model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    plot_training_curve(combine_histories(initial_history, fine_tune_history))
    return final_model_path


def main() -> int:
    try:
        train(parse_args())
    except (FileNotFoundError, ImportError, RuntimeError, ValueError, OSError) as error:
        print(f"Training failed: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())