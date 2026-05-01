#!/usr/bin/env python3
import argparse
import copy
import pickle
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a loader-compatible evaluation dataset root from GT custom data."
    )
    parser.add_argument(
        "--src_root",
        type=str,
        default="data/data/custom",
        help="Source GT dataset root containing points/ and custom_infos_val.pkl",
    )
    parser.add_argument(
        "--dst_root",
        type=str,
        default="data/custom_eval_gt_v1",
        help="Destination evaluation dataset root to create",
    )
    parser.add_argument(
        "--frame_id_map",
        type=str,
        default="data/custom_points_ps_dataset_v2/frame_id_map.csv",
        help="Optional CSV mapping newer frame ids to original point filenames",
    )
    return parser.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def to_plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: to_plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_plain(item) for item in value]
    if isinstance(value, tuple):
        return [to_plain(item) for item in value]
    return value


def load_frame_id_map(csv_path: Path):
    import csv

    if not csv_path.exists():
        return {}

    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows or "frame_id" not in rows[0] or "relative_source_path" not in rows[0]:
        return {}

    return {
        str(row["frame_id"]).zfill(6): Path(row["relative_source_path"]).stem
        for row in rows
    }


def main():
    args = parse_args()
    src_root = Path(args.src_root).expanduser().resolve()
    dst_root = Path(args.dst_root).expanduser().resolve()

    src_infos_path = src_root / "custom_infos_val.pkl"
    src_points_dir = src_root / "points"
    frame_id_map_path = Path(args.frame_id_map).expanduser().resolve()
    if not src_infos_path.exists():
        raise FileNotFoundError(f"Missing source infos: {src_infos_path}")
    if not src_points_dir.exists():
        raise FileNotFoundError(f"Missing source points dir: {src_points_dir}")

    with src_infos_path.open("rb") as f:
        src_infos = pickle.load(f)
    frame_id_map = load_frame_id_map(frame_id_map_path)

    velodyne_dir = dst_root / "training" / "velodyne"
    imagesets_dir = dst_root / "ImageSets"
    ensure_dir(velodyne_dir)
    ensure_dir(imagesets_dir)

    new_infos = []
    frame_ids = []
    skipped_frame_ids = []

    for sample_idx, src_info in enumerate(src_infos):
        info = copy.deepcopy(src_info)
        pc_info = copy.deepcopy(info.get("point_cloud", {}))
        frame_id = str(pc_info["lidar_idx"])
        src_points_path = src_points_dir / f"{frame_id}.npy"
        if not src_points_path.exists() and frame_id in frame_id_map:
            src_points_path = src_points_dir / f"{frame_id_map[frame_id]}.npy"
        if not src_points_path.exists():
            skipped_frame_ids.append(frame_id)
            continue

        points = np.load(src_points_path)
        if points.ndim != 2 or points.shape[1] != 4:
            raise ValueError(f"Expected (N,4) float32 points in {src_points_path}, got {points.shape}")
        points = np.asarray(points, dtype=np.float32)

        dst_points_path = velodyne_dir / f"{frame_id}.bin"
        points.tofile(dst_points_path)

        info["frame_id"] = frame_id
        info["lidar_path"] = str(Path("training") / "velodyne" / f"{frame_id}.bin")
        info["timestamp"] = int(len(new_infos))
        info["pose"] = np.eye(4, dtype=np.float32)
        info["point_cloud"] = {
            "num_features": 4,
            "lidar_idx": frame_id,
            "lidar_sequence": "custom_eval_seq_0",
            "sample_idx": int(len(new_infos)),
        }
        new_infos.append(to_plain(info))
        frame_ids.append(frame_id)

    (imagesets_dir / "train.txt").write_text("")
    (imagesets_dir / "test.txt").write_text("")
    (imagesets_dir / "val.txt").write_text("\n".join(frame_ids) + ("\n" if frame_ids else ""))

    with (dst_root / "custom_infos_val.pkl").open("wb") as f:
        pickle.dump(new_infos, f)
    if skipped_frame_ids:
        (dst_root / "skipped_frame_ids.txt").write_text("\n".join(skipped_frame_ids) + "\n")

    print(f"Source root: {src_root}")
    print(f"Destination root: {dst_root}")
    print(f"Val frames: {len(frame_ids)}")
    print(f"Skipped frames: {len(skipped_frame_ids)}")
    print(f"Saved infos: {dst_root / 'custom_infos_val.pkl'}")
    print(f"Saved velodyne dir: {velodyne_dir}")


if __name__ == "__main__":
    main()
