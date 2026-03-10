import argparse
import open3d as o3d
import numpy as np
import os
from pathlib import Path

# Example usage: python3 scripts/convert_pcd_to_npy.py --source_dir data/custom/location2 --target_dir data/custom/points_location2

def main():
    parser = argparse.ArgumentParser(description="Convert PCD files to NPY format for OpenPCDet.")
    parser.add_argument('--source_dir', type=str, required=True, 
                        help="Path to the source directory containing .pcd files.")
    parser.add_argument('--target_dir', type=str, required=True, 
                        help="Path to the target directory to save .npy files.")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    target_dir = Path(args.target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)

    # Get all .pcd files
    pcd_files = list(source_dir.glob('*.pcd'))
    pcd_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f.stem)) or 0))

    print(f"Found {len(pcd_files)} PCD files. Starting conversion (in order)...")

    for pcd_path in pcd_files:
        pcd = o3d.io.read_point_cloud(str(pcd_path))
        points = np.asarray(pcd.points).astype(np.float32)
        
        # Handle intensity
        # Note: If the PCD file does not contain intensity information, we pad it with 0 
        # to maintain the 4 features required by OpenPCDet.
        num_points = points.shape[0]
        intensity = np.zeros((num_points, 1), dtype=np.float32)
        
        # [x, y, z, intensity]
        point_cloud_npy = np.concatenate([points, intensity], axis=1)

        target_path = target_dir / (pcd_path.stem + '.npy')
        np.save(str(target_path), point_cloud_npy)
        print(f"finish: {target_path.name}")

if __name__ == '__main__':
    main()