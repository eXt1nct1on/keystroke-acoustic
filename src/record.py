"""Record microphone audio and timestamped keyboard events."""

from __future__ import annotations

import argparse
import csv
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:
    np = None

try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    from pynput import keyboard
except ImportError:
    keyboard = None

try:
    import soundfile as sf
except ImportError:
    sf = None


DEFAULT_TARGET_KEYS = list("0123456789")
DEFAULT_REPETITIONS = 3
SAMPLE_RATE = 44_100
CHANNELS = 1


def key_name(key: Any) -> str:
    """Return a stable, readable name for a pynput key."""
    character = getattr(key, "char", None)
    if character is not None:
        return character

    name = getattr(key, "name", None)
    if name is not None:
        return name

    return str(key)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record keystroke audio and timestamped keyboard events."
    )
    parser.add_argument(
        "--keys",
        nargs="+",
        default=DEFAULT_TARGET_KEYS,
        help="Target keys, in the order they should be pressed (default: 0 1 ... 9).",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=DEFAULT_REPETITIONS,
        help="Number of times to repeat the target-key sequence (default: 3).",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=SAMPLE_RATE,
        help="Microphone sample rate in Hz (default: 44100).",
    )
    parser.add_argument(
        "--prompt-pause",
        type=float,
        default=0.2,
        help="Pause between prompts in seconds (default: 0.2).",
    )
    return parser.parse_args()


def save_recording(
    audio_chunks: list[np.ndarray],
    keypresses: list[tuple[str, float]],
    output_directory: Path,
    sample_rate: int,
) -> tuple[Path, Path]:
    """Write the collected audio and keypress events using one matching stem."""
    if sf is None or np is None:
        raise RuntimeError(
            "Saving requires soundfile and numpy. Install dependencies with "
            "'python -m pip install soundfile numpy'."
        )

    output_directory.mkdir(parents=True, exist_ok=True)
    stem = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    audio_path = output_directory / f"{stem}.wav"
    log_path = output_directory / f"{stem}.csv"

    if audio_chunks:
        audio = np.concatenate(audio_chunks, axis=0)
    else:
        audio = np.empty((0, CHANNELS), dtype=np.float32)
    sf.write(audio_path, audio, sample_rate)

    with log_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(("key", "timestamp"))
        writer.writerows(keypresses)

    return audio_path, log_path


def record(
    target_keys: list[str],
    repetitions: int,
    sample_rate: int = SAMPLE_RATE,
    prompt_pause: float = 0.2,
) -> tuple[Path, Path]:
    """Record audio until all targets are pressed or Escape is pressed."""
    if np is None or sd is None or keyboard is None or sf is None:
        missing = []
        if np is None:
            missing.append("numpy")
        if sd is None:
            missing.append("sounddevice")
        if keyboard is None:
            missing.append("pynput")
        if sf is None:
            missing.append("soundfile")
        raise RuntimeError(
            f"Missing required package(s): {', '.join(missing)}. "
            "Install them with 'python -m pip install -r requirements.txt'."
        )
    if not target_keys:
        raise ValueError("At least one target key is required.")
    if repetitions < 1:
        raise ValueError("Repetitions must be at least 1.")
    if sample_rate < 1:
        raise ValueError("Sample rate must be positive.")

    audio_chunks: list[np.ndarray] = []
    keypresses: list[tuple[str, float]] = []
    audio_lock = threading.Lock()
    key_events: queue.Queue[str] = queue.Queue()
    stop_event = threading.Event()
    recording_start = 0.0

    def audio_callback(
        indata: np.ndarray,
        frames: int,
        callback_time: Any,
        status: Any,
    ) -> None:
        if status:
            print(f"Audio warning: {status}", file=sys.stderr)
        with audio_lock:
            audio_chunks.append(indata.copy())

    def on_press(key: Any) -> bool | None:
        name = key_name(key)
        timestamp = time.perf_counter() - recording_start
        keypresses.append((name, timestamp))
        if name == "esc":
            stop_event.set()
            return False
        key_events.put(name)
        return None

    stream = sd.InputStream(
        samplerate=sample_rate,
        channels=CHANNELS,
        dtype="float32",
        callback=audio_callback,
    )
    listener = keyboard.Listener(on_press=on_press)
    completed = False
    stream_started = False
    listener_started = False

    try:
        recording_start = time.perf_counter()
        stream.start()
        stream_started = True
        listener.start()
        listener_started = True
        total_targets = len(target_keys) * repetitions

        for repetition in range(repetitions):
            for target_index, target in enumerate(target_keys):
                if stop_event.is_set():
                    break
                position = repetition * len(target_keys) + target_index + 1
                print(f"Press '{target}' now ({position}/{total_targets})")
                while not stop_event.is_set():
                    try:
                        pressed_key = key_events.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if pressed_key == target:
                        break
                    print(f"Received '{pressed_key}'. Press '{target}' now.")
                time.sleep(prompt_pause)
            if stop_event.is_set():
                break
        completed = not stop_event.is_set()
    finally:
        stop_event.set()
        if listener_started:
            listener.stop()
            listener.join()
        if stream_started:
            stream.stop()
        stream.close()

    output_directory = Path(__file__).resolve().parents[1] / "data" / "raw"
    with audio_lock:
        audio_snapshot = list(audio_chunks)
    audio_path, log_path = save_recording(
        audio_snapshot, keypresses, output_directory, sample_rate
    )
    if completed:
        print("Target sequence complete.")
    else:
        print("Recording stopped early.")
    print(f"Audio saved to {audio_path}")
    print(f"Keypress log saved to {log_path}")
    return audio_path, log_path


def main() -> int:
    args = parse_args()
    try:
        record(args.keys, args.repetitions, args.sample_rate, args.prompt_pause)
    except (RuntimeError, ValueError, OSError) as error:
        print(f"Recording failed: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nRecording interrupted.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())