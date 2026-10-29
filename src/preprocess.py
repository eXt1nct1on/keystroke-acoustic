"""Convert timestamped raw recordings into mel-spectrogram training images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

try:
    from audio_utils import extract_clip, load_audio, save_spectrogram
except ImportError:
    extract_clip = None
    load_audio = None
    save_spectrogram = None

try:
    import numpy as np
except ImportError:
    np = None


DEFAULT_OFFSET_SECONDS = 0.005
DEFAULT_SEED = 42


@dataclass(frozen=True)
class Keypress:
    key: str
    timestamp: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create mel-spectrogram images from raw keystroke recordings."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "raw",
        help="Directory containing matching .wav and .csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "spectrograms",
        help="Directory containing train/<key> and val/<key> outputs.",
    )
    parser.add_argument(
        "--offset",
        type=float,
        default=DEFAULT_OFFSET_SECONDS,
        help="Seconds to shift the initial window after the OS timestamp.",
    )
    parser.add_argument(
        "--refine-onset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Refine the window around the first detected acoustic onset.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Validation fraction applied independently for each key.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed used for reproducible per-key splits.",
    )
    return parser.parse_args()


def safe_key_name(key: str) -> str:
    """Make a CSV key safe to use as a directory and filename component."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", key.strip())
    return cleaned or "unknown"


def load_keypresses(csv_path: Path) -> list[Keypress]:
    keypresses: list[Keypress] = []
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None or not {"key", "timestamp"}.issubset(
            reader.fieldnames
        ):
            raise ValueError("CSV must contain 'key' and 'timestamp' columns.")
        for row_number, row in enumerate(reader, start=2):
            try:
                keypresses.append(
                    Keypress(row["key"], float(row["timestamp"]))
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid keypress row {row_number} in {csv_path.name}: {row}"
                ) from error
    return keypresses


def split_keypresses(
    keypresses: list[Keypress], val_fraction: float, seed: int
) -> dict[str, list[tuple[Keypress, str]]]:
    """Assign each key's events to train/val independently."""
    grouped: dict[str, list[tuple[int, Keypress]]] = defaultdict(list)
    for index, keypress in enumerate(keypresses):
        grouped[keypress.key].append((index, keypress))

    split_events: dict[str, list[tuple[Keypress, str]]] = defaultdict(list)
    for key, indexed_events in grouped.items():
        events = [event for _, event in indexed_events]
        shuffled = list(events)
        stable_key_seed = int.from_bytes(
            hashlib.sha256(key.encode("utf-8")).digest()[:4], "big"
        )
        random.Random(seed + stable_key_seed).shuffle(shuffled)
        validation_count = round(len(shuffled) * val_fraction)
        if len(shuffled) > 1:
            validation_count = max(1, min(validation_count, len(shuffled) - 1))
        else:
            validation_count = 0
        validation_ids = {id(event) for event in shuffled[:validation_count]}
        split_events[key].extend(
            (event, "val" if id(event) in validation_ids else "train")
            for event in events
        )
    return dict(split_events)


def process_recording(
    wav_path: Path,
    csv_path: Path,
    output_dir: Path,
    offset: float,
    refine_onset: bool,
    val_fraction: float,
    seed: int,
) -> dict[str, int]:
    audio, sample_rate = load_audio(wav_path)
    keypresses = load_keypresses(csv_path)
    split_events = split_keypresses(keypresses, val_fraction, seed)
    counts: dict[str, int] = defaultdict(int)

    event_index = 0
    for key, events in split_events.items():
        for event, split in events:
            clip = extract_clip(
                audio, sample_rate, event.timestamp, offset, refine_onset
            )
            filename = f"{wav_path.stem}_{event_index:05d}.png"
            save_spectrogram(
                clip,
                sample_rate,
                output_dir / split / safe_key_name(key) / filename,
            )
            counts[key] += 1
            event_index += 1
    return counts


def process_all(
    raw_dir: Path,
    output_dir: Path,
    offset: float = DEFAULT_OFFSET_SECONDS,
    refine_onset: bool = True,
    val_fraction: float = 0.2,
    seed: int = DEFAULT_SEED,
) -> dict[str, int]:
    if extract_clip is None or load_audio is None or save_spectrogram is None or np is None:
        missing = []
        if extract_clip is None or load_audio is None:
            missing.append("librosa")
        if np is None:
            missing.append("numpy")
        if save_spectrogram is None:
            missing.append("matplotlib/pillow")
        raise RuntimeError(
            f"Missing required package(s): {', '.join(missing)}. "
            "Install them with 'python -m pip install -r requirements.txt'."
        )
    if not 0 <= val_fraction < 1:
        raise ValueError("Validation fraction must be at least 0 and less than 1.")
    if offset < 0:
        raise ValueError("Offset must not be negative.")
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw directory does not exist: {raw_dir}")

    totals: dict[str, int] = defaultdict(int)
    wav_files = sorted(raw_dir.glob("*.wav"))
    for wav_path in wav_files:
        csv_path = wav_path.with_suffix(".csv")
        if not csv_path.is_file():
            print(f"Skipping {wav_path.name}: matching CSV not found.", file=sys.stderr)
            continue
        try:
            counts = process_recording(
                wav_path,
                csv_path,
                output_dir,
                offset,
                refine_onset,
                val_fraction,
                seed,
            )
        except (OSError, ValueError, RuntimeError) as error:
            print(f"Skipping {wav_path.name}: {error}", file=sys.stderr)
            continue
        for key, count in counts.items():
            totals[key] += count
    return dict(totals)


def print_summary(counts: dict[str, int]) -> None:
    print("\nGenerated clips by key")
    print("---------------------")
    if not counts:
        print("No clips generated.")
        return
    print(f"{'Key':<20} {'Clips':>8}  Status")
    for key in sorted(counts):
        count = counts[key]
        status = "NEEDS MORE RECORDING" if count < 20 else "OK"
        print(f"{key:<20} {count:>8}  {status}")


def main() -> int:
    args = parse_args()
    try:
        counts = process_all(
            args.raw_dir,
            args.output_dir,
            args.offset,
            args.refine_onset,
            args.val_fraction,
            args.seed,
        )
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as error:
        print(f"Preprocessing failed: {error}", file=sys.stderr)
        return 1
    print_summary(counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())