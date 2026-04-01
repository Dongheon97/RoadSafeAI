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


## Mar. 24

### Experiments & Evaluation
**Goal:** Execute the first two control groups of the ablation study (Source-Only Baseline vs. Few-Shot Fine-tuning). Please ensure all commands are executed within the `~/RoadSafeAI/tools/` directory.

**Experiment A: Source-Only Baseline**
Directly evaluate the pre-trained KITTI model (which has never seen the construction site data) on our validation set.
* **Expected Result:** Pedestrian AP will be close to 0% (due to the severe Sensor Domain Gap and viewpoint differences).
* **Command:**
```bash
python test.py --cfg_file cfgs/custom_models/pv_rcnn.yaml --batch_size 1 --ckpt ../checkpoints/pv_rcnn_8369.pth
```

**Experiment B: Few-Shot Fine-tuning**
Load the pre-trained weights and fine-tune the model for 80 epochs using our manually annotated 68-frame training set.
* **Command:**
```bash
python train.py --cfg_file cfgs/custom_models/pv_rcnn.yaml --batch_size 2 --epochs 80 --pretrained_model ../checkpoints/pv_rcnn_8369.pth
```

**Experiment C: Few-Shot Evaluation**
Use the newly fine-tuned weights from Experiment B to re-evaluate the 34-frame validation set.
* **Expected Result:** A significant surge in AP (Recall@0.5 reaching over 80%), proving that few-shot fine-tuning can effectively adapt the model to our specific construction site scenario.
* **Command:**
```bash
python test.py --cfg_file cfgs/custom_models/pv_rcnn.yaml --batch_size 1 --ckpt ../output/custom_models/pv_rcnn/default/ckpt/checkpoint_epoch_80.pth
```

---
## Get Dataset

### 2. Download Required Assets
Please download the following three archive files from our shared cloud drive [https://purdue0-my.sharepoint.com/my?id=%2Fpersonal%2Flin2289%5Fpurdue%5Fedu%2FDocuments%2FECE%2060001%20I2I%20Lidar&viewid=72894e5b%2D8d56%2D496f%2Dafbb%2Ded6fbe714f57]:
1. `baseline_checkpoints.tar.gz` (Pre-trained KITTI weights for Source-Only Baseline)
2. `my_trained_output.tar.gz` (Our fine-tuned Few-Shot weights)
3. `custom_dataset.tar.gz` (The processed .npy point clouds and pedestrian labels)

*Place these three `.tar.gz` files directly into the `RoadSafeAI/` root directory.*

### 3. Extract Files to the Correct Directories
Run the following commands in your terminal. The `tar` command will automatically create the required folders and extract the files into their exact original locations:

```bash
# Extract pre-trained baseline weights
tar -xzvf baseline_checkpoints.tar.gz

# Extract our fine-tuned weights (Epoch 80)
tar -xzvf my_trained_output.tar.gz

# Extract the custom dataset
tar -xzvf custom_dataset.tar.gz
```

### 4. Verify Directory Structure
After extraction, your `RoadSafeAI` folder should look exactly like this:

```text
RoadSafeAI/
├── checkpoints/
│   └── pv_rcnn_8369.pth                  <-- Source-Only Baseline Weight
├── data/
│   └── custom/                           <-- Extracted Dataset
│       ├── ImageSets/
│       ├── labels/
│       ├── points/
│       ├── custom_infos_train.pkl
│       └── custom_infos_val.pkl
└── output/
    └── custom_models/
        └── pv_rcnn/
            └── default/
                └── ckpt/
                    └── checkpoint_epoch_80.pth  <-- Few-Shot Fine-tuned Weight
```

### 5. Run Evaluations!
Once the files are in place, you can immediately run the evaluation scripts to reproduce the AP scores.

**Test A: Source-Only Baseline (0% AP expected)**
```bash
cd tools
python test.py --cfg_file cfgs/custom_models/pv_rcnn.yaml --batch_size 1 --ckpt ../checkpoints/pv_rcnn_8369.pth
```

**Test B: Few-Shot Fine-tuned Model (~84% AP expected)**
```bash
python test.py --cfg_file cfgs/custom_models/pv_rcnn.yaml --batch_size 1 --ckpt ../output/custom_models/pv_rcnn/default/ckpt/checkpoint_epoch_80.pth
```