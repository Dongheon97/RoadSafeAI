# RoadSafeAI

# OpenPCDet Custom Dataset Workflow

## Phase 1: PCD to NPY Conversion

**1. Activate Environment**

Ensure you are using the project's Conda environment before running any scripts:

Bash

```
conda activate openpcdet
```

**2. Run Conversion**

Use the following syntax to convert your `.pcd` data into `.npy` format:

Bash

```
python scripts/convert_pcd_to_npy.py --source_dir <input_folder> --target_dir <output_folder>
```

_Example:_

Bash

```
python scripts/convert_pcd_to_npy.py --source_dir data/custom/location2 --target_dir data/custom/points_location2
```

---

## Phase 2: Custom Dataset Inference

**Step 1: Copy the Point Cloud Data**

Copy your converted `.npy` files into the official points directory for the custom dataset.

Bash

```
cp data/custom/location2/*.npy data/custom/points/
```

**Step 2: Create the Validation Index File**

Scan the points folder to generate a numerically sorted list of frame names (without the `.npy` extension) and save it as your validation index.

Bash

```
ls data/custom/points/*.npy | xargs -n 1 basename | sed 's/\.npy//' | sort -V > data/custom/ImageSets/val.txt
```

**Step 3: Create the Training Index File (Dummy)**

Duplicate the validation list to act as a dummy training index, which is required by the dataset generation script.

Bash

```
cp data/custom/ImageSets/val.txt data/custom/ImageSets/train.txt
```

**Step 4: Generate Dataset Info (.pkl files)**

Run the dataset preparation script to generate the `custom_infos_train.pkl` and `custom_infos_val.pkl` index files.

Bash

```
python -m pcdet.datasets.custom.custom_dataset create_custom_infos tools/cfgs/dataset_configs/custom_dataset.yaml
```

**Step 5: Run Batch Inference**

Execute your custom batch inference script to process all frames and output the predicted 3D bounding boxes as `.txt` label files.

Bash

```
python scripts/batch_inference.py \
    --cfg_file tools/cfgs/custom_models/pv_rcnn.yaml \
    --ckpt checkpoints/pv_rcnn_8369.pth \
    --data_path data/custom/points \
    --ext .npy \
    --output_dir data/custom/labels_predicted
```