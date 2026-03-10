#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path

import numpy as np


SUPPORTED_EXTS = {".pcd", ".ply", ".bin", ".npy"}


def natural_key(path: Path):
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", path.name)]


def parse_pcd_header(fp):
    header = {}
    header_size = 0
    while True:
        line = fp.readline()
        if not line:
            break
        header_size += len(line)
        text = line.decode("utf-8", errors="ignore").strip()
        if not text:
            continue
        parts = text.split()
        key = parts[0].upper()
        if key in {"FIELDS", "SIZE", "TYPE", "COUNT"}:
            header[key] = parts[1:]
        elif key in {"WIDTH", "HEIGHT", "POINTS"} and len(parts) > 1:
            header[key] = int(parts[1])
        elif key == "DATA":
            header["DATA"] = parts[1].lower() if len(parts) > 1 else "binary"
            break
    return header, header_size


def load_pcd(path: Path):
    with path.open("rb") as f:
        header, header_size = parse_pcd_header(f)

    fields = [x.lower() for x in header.get("FIELDS", [])]
    if not fields:
        raise ValueError(f"PCD header has no FIELDS: {path}")
    data_mode = header.get("DATA", "binary")
    n_points = header.get("POINTS", None)

    field_idx = {name: idx for idx, name in enumerate(fields)}
    if not {"x", "y", "z"}.issubset(field_idx.keys()):
        raise ValueError(f"PCD must contain x,y,z fields: {path}")
    intensity_name = "intensity" if "intensity" in field_idx else ("reflectance" if "reflectance" in field_idx else None)

    if data_mode == "binary":
        raw = np.fromfile(path, dtype=np.float32, offset=header_size)
        dim = len(fields)
        if raw.size % dim != 0:
            raw = raw[: raw.size - (raw.size % dim)]
        pts = raw.reshape(-1, dim)
    elif data_mode == "ascii":
        with path.open("rb") as f:
            f.seek(header_size)
            pts = np.loadtxt(f, dtype=np.float32)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)
    else:
        raise NotImplementedError(f"Unsupported PCD DATA mode: {data_mode}")

    if n_points is not None and pts.shape[0] > n_points:
        pts = pts[:n_points]

    xyz = pts[:, [field_idx["x"], field_idx["y"], field_idx["z"]]]
    if intensity_name is not None:
        intensity = pts[:, field_idx[intensity_name]].reshape(-1, 1)
    else:
        intensity = np.zeros((xyz.shape[0], 1), dtype=np.float32)
    return np.hstack([xyz, intensity]).astype(np.float32)


def load_ply(path: Path):
    try:
        import open3d as o3d
    except ImportError as e:
        raise ImportError("open3d is required to read .ply files") from e

    pc = o3d.io.read_point_cloud(str(path))
    xyz = np.asarray(pc.points, dtype=np.float32)
    if xyz.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    intensity = np.zeros((xyz.shape[0], 1), dtype=np.float32)
    return np.hstack([xyz, intensity])


def load_bin(path: Path):
    arr = np.fromfile(path, dtype=np.float32)
    for dim in (4, 5, 3):
        if arr.size % dim == 0:
            pts = arr.reshape(-1, dim)
            if dim >= 4:
                return pts[:, :4].astype(np.float32)
            xyz = pts[:, :3].astype(np.float32)
            return np.hstack([xyz, np.zeros((xyz.shape[0], 1), dtype=np.float32)])
    raise ValueError(f"Cannot infer point dimension for .bin file: {path}")


def load_npy(path: Path):
    arr = np.load(path)
    if arr.ndim == 1:
        for dim in (4, 5, 3):
            if arr.size % dim == 0:
                arr = arr.reshape(-1, dim)
                break
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Unsupported .npy point shape for {path}: {arr.shape}")
    xyz = arr[:, :3].astype(np.float32)
    if arr.shape[1] >= 4:
        intensity = arr[:, 3:4].astype(np.float32)
    else:
        intensity = np.zeros((xyz.shape[0], 1), dtype=np.float32)
    return np.hstack([xyz, intensity]).astype(np.float32)


def load_points(path: Path):
    ext = path.suffix.lower()
    if ext == ".pcd":
        return load_pcd(path)
    if ext == ".ply":
        return load_ply(path)
    if ext == ".bin":
        return load_bin(path)
    if ext == ".npy":
        return load_npy(path)
    raise NotImplementedError(f"Unsupported extension: {ext}")


def main():
    parser = argparse.ArgumentParser(description="Convert point clouds to KITTI/OpenPCDet .bin (Nx4 float32).")
    parser.add_argument("--input", type=str, required=True, help="Input raw folder.")
    parser.add_argument("--output", type=str, required=True, help="Output folder (e.g., data/custom_dataset/training/velodyne).")
    parser.add_argument("--start_id", type=int, default=1, help="Starting frame id (default: 1).")
    parser.add_argument("--digits", type=int, default=6, help="Zero-padding width for frame ids.")
    parser.add_argument("--save_mapping", action="store_true", help="Save original-to-frame_id mapping CSV.")
    args = parser.parse_args()

    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    files = sorted(files, key=natural_key)
    if not files:
        raise FileNotFoundError(f"No supported input files found in: {input_dir}")

    mapping_rows = [("frame_id", "relative_source_path", "num_points")]
    cur_id = args.start_id
    for src in files:
        points = load_points(src)
        if points.size == 0:
            continue
        valid_mask = np.isfinite(points).all(axis=1)
        points = points[valid_mask]
        frame_id = f"{cur_id:0{args.digits}d}"
        out_path = output_dir / f"{frame_id}.bin"
        points.astype(np.float32).tofile(out_path)
        mapping_rows.append((frame_id, str(src.relative_to(input_dir)), str(points.shape[0])))
        cur_id += 1

    print(f"Converted frames: {cur_id - args.start_id}")
    print(f"Output dir: {output_dir}")

    if args.save_mapping:
        mapping_path = output_dir.parent.parent / "frame_id_map.csv"
        with mapping_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(mapping_rows)
        print(f"Saved mapping: {mapping_path}")


if __name__ == "__main__":
    main()
