# RoadSafeAI

### PCD to NPY Converter for OpenPCDet Commands

**1. Activate Environment**

Ensure you are using the project's Conda environment:

```
conda activate openpcdet
```

**2. Run Conversion**

Use the following syntax to convert your data:

```
python convert_pcd_to_npy.py --source_dir <input_folder> --target_dir <output_folder>
```

**Example:**

```
python convert_pcd_to_npy.py --source_dir data/custom/location2 --target_dir data/custom/points_location2
```
