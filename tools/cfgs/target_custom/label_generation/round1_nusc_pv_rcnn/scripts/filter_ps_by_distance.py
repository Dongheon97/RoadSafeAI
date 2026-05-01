#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path

import numpy as np


def filter_entry(entry, max_distance):
    gt_boxes = entry.get("gt_boxes")
    if gt_boxes is None:
        return entry

    gt_boxes = np.asarray(gt_boxes)
    if gt_boxes.size == 0:
        return entry

    distances = np.linalg.norm(gt_boxes[:, :3], axis=1)
    mask = distances <= max_distance

    filtered = dict(entry)
    filtered["gt_boxes"] = gt_boxes[mask]

    for key in ("num_pts", "memory_counter"):
        if key in filtered and filtered[key] is not None:
            filtered[key] = np.asarray(filtered[key])[mask]

    return filtered


def main():
    parser = argparse.ArgumentParser(description="Filter pseudo-label boxes by radial distance from the origin.")
    parser.add_argument("--input", required=True, help="Path to input pseudo-label pickle.")
    parser.add_argument("--output", required=True, help="Path to output filtered pseudo-label pickle.")
    parser.add_argument(
        "--max-distance",
        type=float,
        required=True,
        help="Maximum Euclidean distance in meters for keeping a box center.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open("rb") as f:
        ps_dict = pickle.load(f)

    filtered_dict = {
        frame_id: filter_entry(entry, args.max_distance)
        for frame_id, entry in ps_dict.items()
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(filtered_dict, f)

    total_frames = len(filtered_dict)
    total_boxes = sum(np.asarray(entry.get("gt_boxes", [])).shape[0] for entry in filtered_dict.values())
    print(f"Wrote {output_path}")
    print(f"frames={total_frames} boxes={total_boxes} max_distance={args.max_distance}")


if __name__ == "__main__":
    main()
