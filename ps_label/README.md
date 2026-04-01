# `ps_label` Usage Guide

This folder contains a local pseudo-label bundle for the current custom-target MS3D workflow:

- [`ps_label_e0.pkl`](/home/dongheon/workspace/c2i/RoadSafeAI/ps_label/ps_label_e0.pkl)
- [`train.txt`](/home/dongheon/workspace/c2i/RoadSafeAI/ps_label/train.txt)

`ps_label_e0.pkl` is the pseudo-label file used by training. In this repo, it is typically the training-time copy of the file pointed to by `SELF_TRAIN.INIT_PS`, not a separate label format.

## What Each File Means

- [`train.txt`](/home/dongheon/workspace/c2i/RoadSafeAI/ps_label/train.txt)
  Target frame IDs used for training. Each line is a frame ID such as `000001`.
- [`ps_label_e0.pkl`](/home/dongheon/workspace/c2i/RoadSafeAI/ps_label/ps_label_e0.pkl)
  Pseudo labels for those training frames. The top-level structure is a dictionary keyed by frame ID.

Conceptually:

- `train.txt` tells you which LiDAR frames belong to the training split.
- `ps_label_e0.pkl` tells you which pseudo bounding boxes belong to each of those frames.

## Pickle Structure

Observed structure in this repo:

- Top-level type: `dict[str, dict]`
- One entry per frame ID
- Per-frame keys:
  - `gt_boxes`
  - `num_pts`
  - `memory_counter`

`gt_boxes` has shape `(N, 9)` with this field order:

```text
x, y, z, dx, dy, dz, heading, class_id, score
```

Notes:

- Negative `class_id` means an ignore pseudo label.
- `abs(class_id) == 2` means `Pedestrian`.
- Example:
  - `class_id == 2`: positive pedestrian pseudo label
  - `class_id == -2`: ignored pedestrian pseudo label

## Load the Pickle

```python
import pickle

pkl_path = "/home/dongheon/workspace/c2i/RoadSafeAI/ps_label/ps_label_e0.pkl"

with open(pkl_path, "rb") as f:
    ps_dict = pickle.load(f)

print(type(ps_dict))
print(len(ps_dict))
print(next(iter(ps_dict)))
```

## Select One Frame

```python
frame_id = "000001"
entry = ps_dict[frame_id]

print(entry.keys())
boxes = entry["gt_boxes"]           # shape: (N, 9)
num_pts = entry["num_pts"]          # shape: (N,)
memory_counter = entry["memory_counter"]  # shape: (N,)
```

## Extract Bounding Boxes

```python
boxes = entry["gt_boxes"]

xyz = boxes[:, 0:3]
dims = boxes[:, 3:6]
heading = boxes[:, 6]
class_id = boxes[:, 7]
score = boxes[:, 8]
```

## Filter Pedestrian Boxes Only

All pedestrian pseudo labels:

```python
ped_boxes = boxes[abs(boxes[:, 7].astype(int)) == 2]
```

Only positive pedestrian pseudo labels:

```python
ped_pos_boxes = boxes[boxes[:, 7].astype(int) == 2]
```

Only ignored pedestrian pseudo labels:

```python
ped_ign_boxes = boxes[boxes[:, 7].astype(int) == -2]
```

## Match the Pickle to `train.txt`

Use this to confirm that the frame IDs in the pseudo-label file match the training split:

```python
from pathlib import Path
import pickle

train_txt = Path("/home/dongheon/workspace/c2i/RoadSafeAI/ps_label/train.txt")
pkl_path = Path("/home/dongheon/workspace/c2i/RoadSafeAI/ps_label/ps_label_e0.pkl")

frame_ids = [line.strip() for line in train_txt.read_text().splitlines() if line.strip()]

with pkl_path.open("rb") as f:
    ps_dict = pickle.load(f)

missing_in_pkl = [fid for fid in frame_ids if fid not in ps_dict]
extra_in_pkl = [fid for fid in ps_dict.keys() if fid not in set(frame_ids)]

print("train frames:", len(frame_ids))
print("pkl frames:", len(ps_dict))
print("missing_in_pkl:", len(missing_in_pkl))
print("extra_in_pkl:", len(extra_in_pkl))
```

## Visualize One Frame

The repo already includes [`visualize_3d.py`](/home/dongheon/workspace/c2i/RoadSafeAI/tools/visualize_3d.py). You can inspect one pseudo-labeled training frame like this:

```bash
docker exec -it ms3d_training bash -lc "
cd /MS3D/tools
export PYTHONPATH=/MS3D
python visualize_3d.py \
  --cfg_file cfgs/target_custom/ms3d_custom_round1_nusc_pvpp_anchorhead.yaml \
  --ps_pkl /MS3D/ps_label/ps_label_e0.pkl \
  --split train \
  --custom_train_split \
  --idx 0 \
  --ps_score_th 0.10
"
```

This reads the pseudo-label pickle and overlays the boxes for the selected training frame.

## Use It for Training

In this repo, self-training reads pseudo labels from `SELF_TRAIN.INIT_PS` in the training config, for example in [`ms3d_custom_round1_nusc_pvpp_anchorhead.yaml`](/home/dongheon/workspace/c2i/RoadSafeAI/tools/cfgs/target_custom/ms3d_custom_round1_nusc_pvpp_anchorhead.yaml).

During training:

- the file at `SELF_TRAIN.INIT_PS` is loaded at startup
- a copy is saved into the run output directory as `ps_label/ps_label_e0.pkl`

That means `ps_label_e0.pkl` is useful in two ways:

- as the pseudo-label file to inspect directly
- as the snapshot record of the exact pseudo labels consumed by that training run

If you want a training run to use this local file directly, set `SELF_TRAIN.INIT_PS` to:

```text
/MS3D/ps_label/ps_label_e0.pkl
```

assuming the repo is mounted at `/MS3D` inside the container.

## Gotchas

- Frame IDs in `ps_label_e0.pkl` must match the IDs in [`train.txt`](/home/dongheon/workspace/c2i/RoadSafeAI/ps_label/train.txt).
- `ps_label_e0.pkl` is a snapshot copy created when training starts.
- In this MS3D setup, pseudo labels are not refreshed during training.
- If a frame has no usable labels, `gt_boxes` can be empty with shape `(0, 9)`.
- Negative class IDs are ignore labels, not a different class.
