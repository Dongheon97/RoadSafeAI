#!/usr/bin/env python3
import argparse
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np


SUPPORTED_EXTS = {".pcd", ".ply", ".bin", ".npy"}


def natural_key(path: Path):
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", path.name)]


def parse_pcd_header(path: Path):
    header = {}
    with path.open("rb") as f:
        while True:
            line = f.readline()
            if not line:
                break
            try:
                text = line.decode("utf-8").strip()
            except UnicodeDecodeError:
                text = ""
            if not text:
                continue
            parts = text.split()
            key = parts[0].upper()
            if key in {"FIELDS", "SIZE", "TYPE", "COUNT"}:
                header[key] = parts[1:]
            elif key in {"WIDTH", "HEIGHT", "POINTS"}:
                if len(parts) > 1:
                    header[key] = int(parts[1])
            elif key == "DATA":
                header["DATA"] = parts[1].lower() if len(parts) > 1 else "unknown"
                break
    return header


def infer_bin_dim(path: Path):
    num_floats = path.stat().st_size // 4
    for dim in (4, 5, 3):
        if num_floats % dim == 0:
            return dim, num_floats // dim
    return None, None


def infer_npy_dim(path: Path):
    arr = np.load(path, mmap_mode="r")
    if arr.ndim == 1:
        for dim in (4, 5, 3):
            if arr.shape[0] % dim == 0:
                return dim, arr.shape[0] // dim
        return None, None
    if arr.ndim >= 2:
        return arr.shape[-1], arr.shape[0]
    return None, None


def summarize(files, sample_count):
    ext_counter = Counter([p.suffix.lower() for p in files])
    samples = files[: min(sample_count, len(files))]

    dims = []
    point_counts = []
    has_intensity = False
    pcd_fields_counter = Counter()
    data_modes = Counter()

    for path in samples:
        ext = path.suffix.lower()
        if ext == ".pcd":
            header = parse_pcd_header(path)
            fields = [f.lower() for f in header.get("FIELDS", [])]
            if fields:
                pcd_fields_counter[tuple(fields)] += 1
                dims.append(len(fields))
            if "intensity" in fields:
                has_intensity = True
            if "POINTS" in header:
                point_counts.append(header["POINTS"])
            if "DATA" in header:
                data_modes[header["DATA"]] += 1
        elif ext == ".bin":
            dim, npts = infer_bin_dim(path)
            if dim is not None:
                dims.append(dim)
                point_counts.append(npts)
                has_intensity |= dim >= 4
        elif ext == ".npy":
            dim, npts = infer_npy_dim(path)
            if dim is not None:
                dims.append(dim)
                point_counts.append(npts)
                has_intensity |= dim >= 4
        elif ext == ".ply":
            # Minimal inspection without heavyweight dependencies.
            with path.open("rb") as f:
                header_lines = []
                for _ in range(100):
                    line = f.readline()
                    if not line:
                        break
                    text = line.decode("utf-8", errors="ignore").strip()
                    header_lines.append(text)
                    if text == "end_header":
                        break
            vertex_count = None
            vertex_props = []
            in_vertex = False
            for line in header_lines:
                parts = line.split()
                if len(parts) >= 3 and parts[:2] == ["element", "vertex"]:
                    vertex_count = int(parts[2])
                    in_vertex = True
                elif in_vertex and parts[:1] == ["property"] and len(parts) >= 3:
                    vertex_props.append(parts[-1].lower())
                elif parts[:1] == ["element"] and len(parts) >= 2 and parts[1] != "vertex":
                    in_vertex = False
            if vertex_count is not None:
                point_counts.append(vertex_count)
            if vertex_props:
                dims.append(len(vertex_props))
                has_intensity |= ("intensity" in vertex_props or "reflectance" in vertex_props)

    summary = {
        "ext_counter": ext_counter,
        "dims": dims,
        "point_counts": point_counts,
        "has_intensity": has_intensity,
        "pcd_fields_counter": pcd_fields_counter,
        "data_modes": data_modes,
    }
    return summary


def print_summary(root, files, summary, sampled):
    dominant = summary["ext_counter"].most_common(1)[0][0] if summary["ext_counter"] else "unknown"
    pts = np.array(summary["point_counts"], dtype=np.float64) if summary["point_counts"] else np.array([])
    dims = summary["dims"]

    print("Dataset summary")
    print("---------------")
    print(f"root: {root}")
    print(f"files: {len(files)}")
    print(f"sampled_for_stats: {sampled}")
    print(f"format: {dominant}")
    print(f"format_distribution: {dict(summary['ext_counter'])}")
    if dims:
        dim_counter = Counter(dims)
        print(f"point_dimension: {dict(dim_counter)}")
    else:
        print("point_dimension: unknown")
    if pts.size > 0:
        print(
            "points per frame: mean {:.1f}, median {:.1f}, min {}, max {}".format(
                float(pts.mean()), float(np.median(pts)), int(pts.min()), int(pts.max())
            )
        )
    else:
        print("points per frame: unknown")

    if summary["pcd_fields_counter"]:
        common_fields = summary["pcd_fields_counter"].most_common(1)[0][0]
        print(f"channels: {' + '.join(common_fields)}")
        print(f"pcd_data_mode: {dict(summary['data_modes'])}")
    else:
        print("channels: xyz + intensity" if summary["has_intensity"] else "channels: xyz")

    # Heuristic-only coordinate note.
    print("coordinate_system: not explicitly stored (assumed sensor frame x-forward, y-left, z-up)")


def main():
    parser = argparse.ArgumentParser(description="Inspect raw point cloud dataset.")
    parser.add_argument("input_dir", type=str, help="Path to raw dataset directory.")
    parser.add_argument("--sample_count", type=int, default=200, help="Number of files to inspect in detail.")
    args = parser.parse_args()

    root = Path(args.input_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Input dir does not exist: {root}")

    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    files = sorted(files, key=natural_key)
    if not files:
        print("No supported point cloud files found (.pcd/.ply/.bin/.npy).")
        return

    summary = summarize(files, args.sample_count)
    print_summary(root, files, summary, min(args.sample_count, len(files)))


if __name__ == "__main__":
    main()
