#!/usr/bin/env python3
import argparse
import random
from pathlib import Path


def write_split(path: Path, frame_ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for fid in frame_ids:
            f.write(f"{fid}\n")


def main():
    parser = argparse.ArgumentParser(description="Create ImageSets train/val/test split files.")
    parser.add_argument("dataset_root", type=str, help="Dataset root (contains training/velodyne).")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-8:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    root = Path(args.dataset_root).expanduser().resolve()
    velodyne_dir = root / "training" / "velodyne"
    if not velodyne_dir.exists():
        raise FileNotFoundError(f"Velodyne directory not found: {velodyne_dir}")

    frame_ids = sorted([p.stem for p in velodyne_dir.glob("*.bin")])
    if not frame_ids:
        raise FileNotFoundError(f"No .bin files found in: {velodyne_dir}")

    rng = random.Random(args.seed)
    shuffled = frame_ids[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    n_test = n - n_train - n_val

    train_ids = sorted(shuffled[:n_train])
    val_ids = sorted(shuffled[n_train : n_train + n_val])
    test_ids = sorted(shuffled[n_train + n_val : n_train + n_val + n_test])

    imagesets_dir = root / "ImageSets"
    write_split(imagesets_dir / "train.txt", train_ids)
    write_split(imagesets_dir / "val.txt", val_ids)
    write_split(imagesets_dir / "test.txt", test_ids)

    print(f"Dataset root: {root}")
    print(f"Total frames: {n}")
    print(f"train/val/test = {len(train_ids)}/{len(val_ids)}/{len(test_ids)}")
    print(f"Saved: {imagesets_dir / 'train.txt'}")
    print(f"Saved: {imagesets_dir / 'val.txt'}")
    print(f"Saved: {imagesets_dir / 'test.txt'}")


if __name__ == "__main__":
    main()
