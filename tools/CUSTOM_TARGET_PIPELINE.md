# Custom Target-Only MS3D Pipeline (RoadSafeAI)

## 1. Dataset preparation

```bash
cd /home/dongheon/workspace/c2i/RoadSafeAI
python tools/dataset_tools/inspect_dataset.py raw_data/
python tools/dataset_tools/convert_to_kitti_bin.py --input raw_data --output data/custom_dataset/training/velodyne
python tools/dataset_tools/create_imagesets.py data/custom_dataset
python tools/dataset_tools/create_custom_infos.py --dataset data/custom_dataset
```

## 2. Round1 pseudo labels

```bash
docker run --gpus all \
  -v /home/dongheon/workspace/c2i/RoadSafeAI:/MS3D \
  -w /MS3D/tools ms3d:sm86 \
  bash -lc "git config --global --add safe.directory /MS3D && \
            PYTHONPATH=/MS3D bash cfgs/target_custom/label_generation/round1/scripts/generate_ensemble_preds.sh && \
            PYTHONPATH=/MS3D bash cfgs/target_custom/label_generation/round1/scripts/run_ms3d.sh"
```

Output pseudo labels:

- `tools/cfgs/target_custom/label_generation/round1/ps_labels/final_ps_dict.pkl`

## 3. Target-only training (pretrained + custom target)

Target-only config:

- `tools/cfgs/target_custom/ms3d_custom_round1_pv_rcnn_plusplus_anchorhead.yaml`

Training:

```bash
docker run --gpus all \
  -v /home/dongheon/workspace/c2i/RoadSafeAI:/MS3D \
  -w /MS3D/tools ms3d:sm86 \
  bash -lc "git config --global --add safe.directory /MS3D && \
            PYTHONPATH=/MS3D python train.py \
              --cfg_file cfgs/target_custom/ms3d_custom_round1_pv_rcnn_plusplus_anchorhead.yaml \
              --extra_tag round1_target_only_smoke \
              --pretrained_model /MS3D/pcdet/models/lyft_uda_pv_rcnn_plusplus_resnet_anchorhead_1xyz_allcls.pth \
              --set OPTIMIZATION.NUM_EPOCHS 1 OPTIMIZATION.BATCH_SIZE_PER_GPU 1"
```

Checkpoint:

- `output/target_custom/ms3d_custom_round1_pv_rcnn_plusplus_anchorhead/round1_target_only_smoke/ckpt/checkpoint_epoch_1.pth`

## 4. Test / inference

```bash
docker run --gpus all \
  -v /home/dongheon/workspace/c2i/RoadSafeAI:/MS3D \
  -w /MS3D/tools ms3d:sm86 \
  bash -lc "git config --global --add safe.directory /MS3D && \
            PYTHONPATH=/MS3D python test.py \
              --cfg_file cfgs/target_custom/ms3d_custom_round1_pv_rcnn_plusplus_anchorhead.yaml \
              --ckpt /MS3D/output/target_custom/ms3d_custom_round1_pv_rcnn_plusplus_anchorhead/round1_target_only_smoke/ckpt/checkpoint_epoch_1.pth \
              --eval_tag retest_round1_target_only \
              --set MODEL.POST_PROCESSING.EVAL_METRIC none DATA_CONFIG.DATA_SPLIT.test val"
```

Result directory:

- `output/target_custom/ms3d_custom_round1_pv_rcnn_plusplus_anchorhead/default/eval/epoch_1/val/retest_round1_target_only/`

Notes:

- Custom target dataset is unlabeled, so evaluation metric is skipped (`EVAL_METRIC: none`).
- `py3d7ada9` suffix in checkpoint version indicates patched RoadSafeAI commit state.

## 5. Separate nuScenes round1 flow (`round1_nusc_pvpp`)

Use this flow when you want to keep the existing `round1` pseudo labels intact and generate a new pseudo-label set from the nuScenes PV-RCNN++ anchorhead pretrained checkpoint.

### 5.1 Detection result generation

```bash
docker exec ms3d_training bash -lc "
cd /MS3D/tools
export PYTHONPATH=/MS3D

python test.py \
  --cfg_file cfgs/nuscenes_models/uda_pv_rcnn_plusplus_resnet_anchorhead.yaml \
  --ckpt /MS3D/pcdet/models/nuscenes_uda_pv_rcnn_plusplus_resnet_anchorhead_10xyzt_allcls.pth \
  --eval_tag nusc10xyzt_custom1xyzt_notta_pvpp_anchor \
  --target_dataset custom \
  --sweeps 1 \
  --batch_size 4 \
  --use_tta 0 \
  --set DATA_CONFIG_TAR.DATA_SPLIT.test train MODEL.POST_PROCESSING.EVAL_METRIC none MODEL.POST_PROCESSING.SCORE_THRESH 0.01
"
```

Expected detection output:

- `/MS3D/output/nuscenes_models/uda_pv_rcnn_plusplus_resnet_anchorhead/default/eval/epoch_10/train/nusc10xyzt_custom1xyzt_notta_pvpp_anchor/result.pkl`

### 5.2 Pseudo-label generation in separate folder

```bash
docker exec ms3d_training bash -lc "
cd /MS3D/tools
export PYTHONPATH=/MS3D:/MS3D/tracker

bash cfgs/target_custom/label_generation/round1_nusc_pvpp/scripts/run_ms3d.sh
"
```

Pseudo-label output folder:

- `/MS3D/tools/cfgs/target_custom/label_generation/round1_nusc_pvpp/ps_labels/`

Raw pseudo-label file:

- `/MS3D/tools/cfgs/target_custom/label_generation/round1_nusc_pvpp/ps_labels/final_ps_dict.pkl`

### 5.3 Pedestrian-only pseudo-label filtering

```bash
docker exec ms3d_training bash -lc "
cd /MS3D/tools
python cfgs/target_custom/label_generation/round1_nusc_pvpp/scripts/filter_pedestrian_ps.py \
  --input /MS3D/tools/cfgs/target_custom/label_generation/round1_nusc_pvpp/ps_labels/final_ps_dict.pkl \
  --output /MS3D/tools/cfgs/target_custom/label_generation/round1_nusc_pvpp/ps_labels/final_ps_dict_ped_only.pkl
"
```

Filtered pseudo-label file:

- `/MS3D/tools/cfgs/target_custom/label_generation/round1_nusc_pvpp/ps_labels/final_ps_dict_ped_only.pkl`

### 5.4 Smoke training with pedestrian-only pseudo labels

```bash
docker exec ms3d_training bash -lc "
cd /MS3D/tools
export PYTHONPATH=/MS3D

python train.py \
  --cfg_file cfgs/target_custom/ms3d_custom_round1_nusc_pvpp_anchorhead.yaml \
  --extra_tag nusc_round1_nusc_pvpp_smoke \
  --pretrained_model /MS3D/pcdet/models/nuscenes_uda_pv_rcnn_plusplus_resnet_anchorhead_10xyzt_allcls.pth \
  --custom_target_scenes \
  --set OPTIMIZATION.NUM_EPOCHS 1 OPTIMIZATION.BATCH_SIZE_PER_GPU 1
"
```

### 5.5 Full training with pedestrian-only pseudo labels

```bash
docker exec ms3d_training bash -lc "
cd /MS3D/tools
export PYTHONPATH=/MS3D

python train.py \
  --cfg_file cfgs/target_custom/ms3d_custom_round1_nusc_pvpp_anchorhead.yaml \
  --extra_tag nusc_round1_nusc_pvpp_full \
  --pretrained_model /MS3D/pcdet/models/nuscenes_uda_pv_rcnn_plusplus_resnet_anchorhead_10xyzt_allcls.pth \
  --custom_target_scenes
"
```

## 6. Optional W&B Logging

If you want online W&B logging, install `wandb` inside the container and provide the required environment variables.

Install:

```bash
docker exec ms3d_training bash -lc "python -m pip install wandb"
```

Example environment:

```bash
export WANDB_API_KEY=<your_api_key>
export WANDB_PROJECT=RoadSafeAI-MS3D
export WANDB_ENTITY=<optional_entity>
export WANDB_RUN_NAME=nusc_round1_nusc_pvpp_full
export WANDB_TAGS=ms3d,custom,nuscenes,pedestrian
export WANDB_MODE=online
```

Training will continue normally if `wandb` is not installed or if `WANDB_PROJECT` is not set.
