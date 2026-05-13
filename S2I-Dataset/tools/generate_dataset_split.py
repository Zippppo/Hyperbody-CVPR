#!/usr/bin/env python3
"""Generate train/val/test split JSON for S2I-Dataset.

The training code joins `data_dir` with entries from the split JSON, so this
script writes plain filenames such as `S2I_00001.npz` by default.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from datetime import datetime
from pathlib import Path


DATASET_ROOT = Path(__file__).resolve().parents[1]
S2I_NAME_RE = re.compile(r"S2I_(\d{5})\.npz$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate S2I-Dataset/info/dataset_split.json from data/S2I_*.npz"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing S2I_*.npz files, relative to S2I-Dataset by default.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("info/dataset_split.json"),
        help="Output split JSON path, relative to S2I-Dataset by default.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling.")
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
        help="Validation ratio. Ignored when --val-count is set.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.05,
        help="Test ratio. Ignored when --test-count is set.",
    )
    parser.add_argument(
        "--val-count",
        type=int,
        default=None,
        help="Fixed validation count. Overrides --val-ratio.",
    )
    parser.add_argument(
        "--test-count",
        type=int,
        default=None,
        help="Fixed test count. Overrides --test-ratio.",
    )
    parser.add_argument(
        "--pattern",
        default="S2I_*.npz",
        help="Glob pattern for dataset files inside --data-dir.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output if it already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned split without writing the JSON file.",
    )
    return parser.parse_args()


def resolve_under_dataset(path: Path) -> Path:
    return path if path.is_absolute() else DATASET_ROOT / path


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(DATASET_ROOT))
    except ValueError:
        return str(path)


def numeric_s2i_id(path: Path) -> int:
    match = S2I_NAME_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(
            f"Unexpected filename {path.name!r}; expected format like S2I_00001.npz"
        )
    return int(match.group(1))


def collect_files(data_dir: Path, pattern: str) -> tuple[list[Path], list[int]]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Data path is not a directory: {data_dir}")

    files = sorted(data_dir.glob(pattern), key=numeric_s2i_id)
    if not files:
        raise FileNotFoundError(f"No files matched {pattern!r} in {data_dir}")

    ids = [numeric_s2i_id(path) for path in files]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate numeric S2I ids found.")
    return files, ids


def compute_split_counts(
    total: int,
    val_ratio: float,
    test_ratio: float,
    val_count: int | None,
    test_count: int | None,
) -> tuple[int, int, int]:
    if total <= 0:
        raise ValueError("total must be positive")
    if val_ratio < 0 or test_ratio < 0:
        raise ValueError("val-ratio and test-ratio must be non-negative")
    if val_ratio + test_ratio >= 1.0 and (val_count is None or test_count is None):
        raise ValueError("val-ratio + test-ratio must be < 1.0")

    resolved_val_count = val_count if val_count is not None else int(total * val_ratio)
    resolved_test_count = test_count if test_count is not None else int(total * test_ratio)

    if resolved_val_count < 0 or resolved_test_count < 0:
        raise ValueError("val-count and test-count must be non-negative")
    if resolved_val_count + resolved_test_count >= total:
        raise ValueError(
            "val_count + test_count must be smaller than the number of dataset files"
        )

    train_count = total - resolved_val_count - resolved_test_count
    return train_count, resolved_val_count, resolved_test_count


def build_split(args: argparse.Namespace) -> dict:
    data_dir = resolve_under_dataset(args.data_dir)
    files, ids = collect_files(data_dir, args.pattern)

    missing_ids = []
    if ids:
        id_set = set(ids)
        missing_ids = [idx for idx in range(min(ids), max(ids) + 1) if idx not in id_set]

    train_count, val_count, test_count = compute_split_counts(
        total=len(files),
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        val_count=args.val_count,
        test_count=args.test_count,
    )

    filenames = [path.name for path in files]
    rng = random.Random(args.seed)
    shuffled = filenames.copy()
    rng.shuffle(shuffled)

    train_files = shuffled[:train_count]
    val_files = shuffled[train_count : train_count + val_count]
    test_files = shuffled[train_count + val_count :]

    output = resolve_under_dataset(args.output)
    return {
        "split_info": {
            "total_samples": len(files),
            "train_count": len(train_files),
            "val_count": len(val_files),
            "test_count": len(test_files),
            "random_seed": args.seed,
            "split_date": datetime.now().strftime("%Y-%m-%d"),
            "data_directory": display_path(data_dir),
            "filename_pattern": args.pattern,
            "id_min": min(ids),
            "id_max": max(ids),
            "missing_numeric_id_count": len(missing_ids),
            "missing_numeric_ids_preview": missing_ids[:20],
            "output_file": display_path(output),
        },
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }


def write_split(split_data: dict, output: Path, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output}. Pass --overwrite to replace it."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(split_data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main() -> None:
    args = parse_args()
    output = resolve_under_dataset(args.output)
    split_data = build_split(args)
    info = split_data["split_info"]

    print("S2I split summary:")
    print(f"  Data dir: {resolve_under_dataset(args.data_dir)}")
    print(f"  Total:    {info['total_samples']}")
    print(f"  Train:    {info['train_count']}")
    print(f"  Val:      {info['val_count']}")
    print(f"  Test:     {info['test_count']}")
    print(f"  ID range: {info['id_min']}..{info['id_max']}")
    print(f"  Missing numeric IDs: {info['missing_numeric_id_count']}")

    if args.dry_run:
        print(f"Dry run only; no file written. Target output: {output}")
        return

    write_split(split_data, output, overwrite=args.overwrite)
    print(f"Saved split JSON to: {output}")


if __name__ == "__main__":
    main()
