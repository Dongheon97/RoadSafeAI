#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path

import numpy as np


DEFAULT_THRESHOLDS = (0.01, 0.05, 0.10, 0.15, 0.20)


def load_result(path: Path):
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list of prediction dicts in {path}, got {type(data).__name__}")
    return data


def to_numpy(array_like, dtype=None):
    if array_like is None:
        return np.array([], dtype=dtype)
    arr = np.asarray(array_like)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def summarize_result(det_annos, focus_class_name, thresholds):
    total_boxes = 0
    nonempty_frames = 0
    class_counts = {}
    threshold_counts = {th: 0 for th in thresholds}
    focus_total = 0
    focus_nonempty_frames = 0
    focus_threshold_counts = {th: 0 for th in thresholds}

    for anno in det_annos:
        scores = to_numpy(anno.get("score"), dtype=float)
        names = to_numpy(anno.get("name"))
        num_boxes = int(scores.shape[0])

        total_boxes += num_boxes
        if num_boxes > 0:
            nonempty_frames += 1

        for th in thresholds:
            threshold_counts[th] += int((scores >= th).sum())

        if names.size > 0:
            names = names.astype(str)
            for name in names:
                class_counts[name] = class_counts.get(name, 0) + 1
            focus_mask = names == focus_class_name
            focus_scores = scores[focus_mask]
            focus_count = int(focus_mask.sum())
            focus_total += focus_count
            if focus_count > 0:
                focus_nonempty_frames += 1
            for th in thresholds:
                focus_threshold_counts[th] += int((focus_scores >= th).sum())

    frame_count = len(det_annos)
    avg_boxes = total_boxes / frame_count if frame_count else 0.0
    avg_focus_boxes = focus_total / frame_count if frame_count else 0.0

    return {
        "frames": frame_count,
        "total_boxes": total_boxes,
        "avg_boxes_per_frame": avg_boxes,
        "nonempty_frames": nonempty_frames,
        "focus_class_name": focus_class_name,
        "focus_boxes": focus_total,
        "avg_focus_boxes_per_frame": avg_focus_boxes,
        "focus_nonempty_frames": focus_nonempty_frames,
        "class_counts": class_counts,
        "threshold_counts": threshold_counts,
        "focus_threshold_counts": focus_threshold_counts,
    }


def print_summary(label, summary):
    print(f"[{label}]")
    print(f"frames={summary['frames']}")
    print(f"total_boxes={summary['total_boxes']}")
    print(f"avg_boxes_per_frame={summary['avg_boxes_per_frame']:.4f}")
    print(f"nonempty_frames={summary['nonempty_frames']}")
    print(f"{summary['focus_class_name']}_boxes={summary['focus_boxes']}")
    print(f"avg_{summary['focus_class_name']}_per_frame={summary['avg_focus_boxes_per_frame']:.4f}")
    print(f"{summary['focus_class_name']}_nonempty_frames={summary['focus_nonempty_frames']}")
    print("threshold_counts_all:")
    for th, count in summary["threshold_counts"].items():
        print(f"  >= {th:.2f}: {count}")
    print(f"threshold_counts_{summary['focus_class_name']}:")
    for th, count in summary["focus_threshold_counts"].items():
        print(f"  >= {th:.2f}: {count}")
    if summary["class_counts"]:
        print("class_counts:")
        for name in sorted(summary["class_counts"]):
            print(f"  {name}: {summary['class_counts'][name]}")
    print()


def print_delta(lhs_label, lhs, rhs_label, rhs):
    print(f"[delta: {rhs_label} - {lhs_label}]")
    print(f"frames={rhs['frames'] - lhs['frames']}")
    print(f"total_boxes={rhs['total_boxes'] - lhs['total_boxes']}")
    print(f"avg_boxes_per_frame={rhs['avg_boxes_per_frame'] - lhs['avg_boxes_per_frame']:.4f}")
    print(f"nonempty_frames={rhs['nonempty_frames'] - lhs['nonempty_frames']}")
    print(f"{rhs['focus_class_name']}_boxes={rhs['focus_boxes'] - lhs['focus_boxes']}")
    print(
        f"avg_{rhs['focus_class_name']}_per_frame="
        f"{rhs['avg_focus_boxes_per_frame'] - lhs['avg_focus_boxes_per_frame']:.4f}"
    )
    print(f"{rhs['focus_class_name']}_nonempty_frames={rhs['focus_nonempty_frames'] - lhs['focus_nonempty_frames']}")
    print("threshold_counts_all:")
    for th in lhs["threshold_counts"]:
        print(f"  >= {th:.2f}: {rhs['threshold_counts'][th] - lhs['threshold_counts'][th]}")
    print(f"threshold_counts_{rhs['focus_class_name']}:")
    for th in lhs["focus_threshold_counts"]:
        print(f"  >= {th:.2f}: {rhs['focus_threshold_counts'][th] - lhs['focus_threshold_counts'][th]}")
    all_names = sorted(set(lhs["class_counts"]) | set(rhs["class_counts"]))
    if all_names:
        print("class_counts:")
        for name in all_names:
            lhs_count = lhs["class_counts"].get(name, 0)
            rhs_count = rhs["class_counts"].get(name, 0)
            print(f"  {name}: {rhs_count - lhs_count}")


def main():
    parser = argparse.ArgumentParser(description="Compare text stats from two OpenPCDet result.pkl files.")
    parser.add_argument("--baseline", required=True, help="Path to baseline result.pkl")
    parser.add_argument("--candidate", required=True, help="Path to candidate result.pkl")
    parser.add_argument("--baseline-label", default="baseline", help="Label for baseline output")
    parser.add_argument("--candidate-label", default="candidate", help="Label for candidate output")
    parser.add_argument("--focus-class", default="Pedestrian", help="Class name to report separately")
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=list(DEFAULT_THRESHOLDS),
        help="Score thresholds to report",
    )
    args = parser.parse_args()

    baseline_path = Path(args.baseline).expanduser().resolve()
    candidate_path = Path(args.candidate).expanduser().resolve()

    baseline = summarize_result(load_result(baseline_path), args.focus_class, args.thresholds)
    candidate = summarize_result(load_result(candidate_path), args.focus_class, args.thresholds)

    print(f"baseline_path={baseline_path}")
    print(f"candidate_path={candidate_path}")
    print()
    print_summary(args.baseline_label, baseline)
    print_summary(args.candidate_label, candidate)
    print_delta(args.baseline_label, baseline, args.candidate_label, candidate)


if __name__ == "__main__":
    main()
