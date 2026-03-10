#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path

import numpy as np


def read_split_ids(split_path: Path):
    if not split_path.exists():
        return []
    return [x.strip() for x in split_path.read_text().splitlines() if x.strip()]


def count_points(bin_path: Path):
    num_floats = bin_path.stat().st_size // 4
    if num_floats % 4 != 0:
        raise ValueError(f"Invalid bin size (not divisible by 4 floats): {bin_path}")
    return num_floats // 4


def build_frame_info(
    frame_id,
    sample_idx,
    pose,
    rel_lidar_path,
    num_points,
    sequence_name,
):
    return {
        "frame_id": frame_id,
        "timestamp": int(sample_idx),
        "point_cloud": {
            "num_features": 4,
            "lidar_idx": frame_id,
            "lidar_sequence": sequence_name,
            "sample_idx": int(sample_idx),
        },
        "lidar_path": rel_lidar_path,
        "num_points": int(num_points),
        "pose": pose.astype(np.float32),
        # Unlabeled target dataset (as requested)
        "annos": None,
    }


def main():
    parser = argparse.ArgumentParser(description="Create custom_infos_*.pkl for flat custom dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset root (e.g., data/custom_dataset).")
    parser.add_argument("--odom", type=str, default=None, help="Optional odometry .npy with shape (N,4,4).")
    parser.add_argument("--sequence_name", type=str, default="custom_seq_0", help="Sequence identifier for metadata.")
    args = parser.parse_args()

    root = Path(args.dataset).expanduser().resolve()
    velodyne_dir = root / "training" / "velodyne"
    imagesets_dir = root / "ImageSets"
    if not velodyne_dir.exists():
        raise FileNotFoundError(f"Missing: {velodyne_dir}")
    if not imagesets_dir.exists():
        raise FileNotFoundError(f"Missing: {imagesets_dir}")

    all_frame_ids = sorted([p.stem for p in velodyne_dir.glob("*.bin")])
    if not all_frame_ids:
        raise FileNotFoundError(f"No .bin files in: {velodyne_dir}")

    train_ids = read_split_ids(imagesets_dir / "train.txt")
    val_ids = read_split_ids(imagesets_dir / "val.txt")
    test_ids = read_split_ids(imagesets_dir / "test.txt")

    frame_set = set(all_frame_ids)
    for split_name, split_ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        missing = [fid for fid in split_ids if fid not in frame_set]
        if missing:
            raise ValueError(f"{split_name}.txt contains missing frame IDs: {missing[:5]}")

    id_to_idx = {fid: idx for idx, fid in enumerate(all_frame_ids)}

    odom = None
    if args.odom:
        odom_path = Path(args.odom).expanduser().resolve()
        odom = np.load(odom_path)
        if odom.ndim != 3 or odom.shape[1:] != (4, 4):
            raise ValueError(f"Odometry must have shape (N,4,4), got {odom.shape}")
        if odom.shape[0] < len(all_frame_ids):
            raise ValueError(
                f"Odometry length ({odom.shape[0]}) is smaller than frame count ({len(all_frame_ids)})"
            )

    def make_infos(split_ids):
        infos = []
        for fid in sorted(split_ids):
            sample_idx = id_to_idx[fid]
            bin_path = velodyne_dir / f"{fid}.bin"
            pose = odom[sample_idx] if odom is not None else np.eye(4, dtype=np.float32)
            rel_path = str(Path("training") / "velodyne" / f"{fid}.bin")
            npts = count_points(bin_path)
            infos.append(
                build_frame_info(
                    frame_id=fid,
                    sample_idx=sample_idx,
                    pose=pose,
                    rel_lidar_path=rel_path,
                    num_points=npts,
                    sequence_name=args.sequence_name,
                )
            )
        return infos

    train_infos = make_infos(train_ids)
    val_infos = make_infos(val_ids)

    with (root / "custom_infos_train.pkl").open("wb") as f:
        pickle.dump(train_infos, f)
    with (root / "custom_infos_val.pkl").open("wb") as f:
        pickle.dump(val_infos, f)

    print(f"Dataset root: {root}")
    print(f"Frames total/train/val/test: {len(all_frame_ids)}/{len(train_ids)}/{len(val_ids)}/{len(test_ids)}")
    print(f"Saved: {root / 'custom_infos_train.pkl'} ({len(train_infos)} infos)")
    print(f"Saved: {root / 'custom_infos_val.pkl'} ({len(val_infos)} infos)")
    if odom is None:
        print("Odometry: not provided, used identity poses")
    else:
        print(f"Odometry: loaded from {args.odom}")


if __name__ == "__main__":
    main()
