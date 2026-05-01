#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import _init_path  # noqa: F401
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import __all__ as dataset_registry
from pcdet.utils import common_utils


def save_bev(points, out_path, title):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.scatter(points[:, 0], points[:, 1], s=0.2, c=points[:, 2], cmap="viridis", alpha=0.7, linewidths=0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Sanity-check custom dataset loading.")
    parser.add_argument("--cfg_file", type=str, required=True, help="Dataset config yaml.")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples for stats.")
    parser.add_argument("--sample_idx", type=int, default=0, help="Index for BEV visualization.")
    parser.add_argument(
        "--bev_out",
        type=str,
        default="tools/custom_dataset_bev.png",
        help="Output image path for BEV visualization.",
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    dataset_cfg = cfg
    cfg_path = Path(args.cfg_file).resolve()
    data_path = Path(dataset_cfg.DATA_PATH)
    if not data_path.is_absolute():
        cwd_candidate = (Path.cwd() / data_path).resolve()
        cfg_candidate = (cfg_path.parent / data_path).resolve()
        dataset_cfg.DATA_PATH = str(cwd_candidate if cwd_candidate.exists() else cfg_candidate)

    logger = common_utils.create_logger()
    dataset_cls = dataset_registry[dataset_cfg.DATASET]
    dataset = dataset_cls(
        dataset_cfg=dataset_cfg,
        class_names=dataset_cfg.CLASS_NAMES,
        training=False,
        root_path=Path(dataset_cfg.DATA_PATH),
        logger=logger,
    )

    n = len(dataset)
    if n == 0:
        print("Dataset is empty.")
        return

    print("Custom Dataset Stats")
    print("--------------------")
    print(f"cfg_file: {args.cfg_file}")
    print(f"dataset_path: {Path(dataset_cfg.DATA_PATH).resolve()}")
    print(f"num_frames: {n}")

    check_count = min(args.num_samples, n)
    num_points = []
    for i in range(check_count):
        raw = dataset.get_lidar(dataset.infos[i]["lidar_path"])
        num_points.append(raw.shape[0])

    num_points = np.asarray(num_points, dtype=np.float32)
    print(f"sampled_frames_for_stats: {check_count}")
    print(
        "points_per_frame: mean {:.1f}, median {:.1f}, min {}, max {}".format(
            float(np.mean(num_points)),
            float(np.median(num_points)),
            int(np.min(num_points)),
            int(np.max(num_points)),
        )
    )

    vis_idx = max(0, min(args.sample_idx, n - 1))
    data = dataset[vis_idx]
    pts = data["points"]
    out_path = Path(args.bev_out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_bev(pts, out_path, f"frame_id={data['frame_id']}")
    print(f"bev_saved: {out_path}")


if __name__ == "__main__":
    main()
