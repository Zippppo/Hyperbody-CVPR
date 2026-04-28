"""Build train/val/test split for S2I_Dataset from S2I_Dataset/train/*.npz."""

import argparse
import glob
import json
import os
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="S2I_Dataset")
    parser.add_argument("--src-subdir", default="train")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    pattern = os.path.join(args.data_root, args.src_subdir, "*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no .npz under {pattern}")

    rel_paths = [os.path.relpath(path, args.data_root) for path in files]
    rng = random.Random(args.seed)
    rng.shuffle(rel_paths)

    total = len(rel_paths)
    n_val = int(total * args.val_ratio)
    n_test = int(total * args.test_ratio)
    val = rel_paths[:n_val]
    test = rel_paths[n_val:n_val + n_test]
    train = rel_paths[n_val + n_test:]

    output = args.output or os.path.join(args.data_root, "dataset_split.json")
    if os.path.exists(output) and not args.overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite to replace it")

    payload = {
        "split_info": {
            "total_samples": total,
            "train_count": len(train),
            "val_count": len(val),
            "test_count": len(test),
            "random_seed": args.seed,
            "source_subdir": args.src_subdir,
        },
        "train": train,
        "val": val,
        "test": test,
    }
    with open(output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {output}: train={len(train)} val={len(val)} test={len(test)}")


if __name__ == "__main__":
    main()
