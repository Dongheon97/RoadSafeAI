# RoadSafeAI

RoadSafeAI is the Stage 1 MS3D adaptation workspace for the custom Hesai LiDAR target domain.

This repository is documented for the `custom` pipeline only:

- target dataset: `custom_points_ps_dataset_v2`
- pseudo-label flow: `round1_nusc_pv_rcnn`
- training config: `tools/cfgs/target_custom/ms3d_custom_round1_pv_rcnn_plusplus_points_nusc.yaml`
- training/inference entrypoints: `tools/train.py`, `tools/test.py`

Public dataset configs remain in the repo, but this README does not cover them.

## What Is Not Tracked

The repository intentionally does not store large or generated artifacts:

- raw data and converted frame payloads under `data/`
- training/eval/video outputs under `output/`
- shared archives under `shared/` and `shared.zip`
- pretrained checkpoints under `pcdet/models/*.pth`
- pseudo-label pickle outputs under `tools/cfgs/target_custom/**/ps_labels/`

Place pretrained checkpoints manually at:

- `pcdet/models/nuscenes_uda_pv_rcnn_plusplus_resnet_anchorhead_10xyzt_allcls.pth`
- `pcdet/models/lyft_uda_pv_rcnn_plusplus_resnet_anchorhead_1xyz_allcls.pth`

## Dataset Layout

The tracked repo keeps only the directory skeleton:

```text
data/
  custom_dataset/
    ImageSets/
      train.txt
      val.txt
      test.txt
    training/
      velodyne/
  custom_points_ps_dataset_v2/
    ImageSets/
      train.txt
      val.txt
      test.txt
    training/
      velodyne/
  custom_eval_gt_v1/
    ImageSets/
      train.txt
      val.txt
      test.txt
    training/
      velodyne/
```

Usage:

- `custom_points_ps_dataset_v2`: primary MS3D target-only training root
- `custom_eval_gt_v1`: optional GT evaluation root built from labeled external data
- `custom_dataset`: reserved flat OpenPCDet-style custom dataset root

The `velodyne` directory name must stay unchanged for OpenPCDet compatibility.

## Docker Workflow

`conda_setup.sh` is removed. Build and run the project through Docker only.

Build the image:

```bash
cd /path/to/RoadSafeAI
docker build -t ms3d:sm86 .
```

Start the working container:

```bash
docker run --gpus all -d \
  --name ms3d_training \
  -v /path/to/RoadSafeAI:/MS3D \
  -w /MS3D/tools \
  ms3d:sm86 \
  tail -f /dev/null
```

Quick verification:

```bash
docker exec ms3d_training bash -lc "python -c 'import torch, spconv, open3d, wandb, SharedArray; print(torch.__version__, torch.cuda.is_available())'"
docker exec ms3d_training bash -lc "nvcc --version"
```

All commands below assume:

```bash
docker exec ms3d_training bash -lc "
cd /MS3D/tools
export PYTHONPATH=/MS3D:/MS3D/tracker
..."
```

## Dataset Preprocessing

Supported raw input formats for `convert_to_kitti_bin.py`:

- `.pcd`
- `.ply`
- `.bin`
- `.npy`

Prepare the primary target root:

```bash
python tools/dataset_tools/inspect_dataset.py raw_data/
python tools/dataset_tools/convert_to_kitti_bin.py \
  --input raw_data \
  --output data/custom_points_ps_dataset_v2/training/velodyne \
  --save_mapping
python tools/dataset_tools/create_imagesets.py data/custom_points_ps_dataset_v2
python tools/dataset_tools/create_custom_infos.py \
  --dataset data/custom_points_ps_dataset_v2 \
  --sequence_name custom_seq_0
```

Build the optional GT evaluation root from labeled external data:

```bash
python tools/dataset_tools/build_custom_eval_gt_root.py \
  --src_root data/data/custom \
  --dst_root data/custom_eval_gt_v1 \
  --frame_id_map data/custom_points_ps_dataset_v2/frame_id_map.csv
```

## Pseudo-Label Generation

Generate pretrained detections:

```bash
docker exec ms3d_training bash -lc "
cd /MS3D/tools
export PYTHONPATH=/MS3D:/MS3D/tracker
bash cfgs/target_custom/label_generation/round1_nusc_pv_rcnn/scripts/generate_ensemble_preds.sh
"
```

Run MS3D label refinement:

```bash
docker exec ms3d_training bash -lc "
cd /MS3D/tools
export PYTHONPATH=/MS3D:/MS3D/tracker
bash cfgs/target_custom/label_generation/round1_nusc_pv_rcnn/scripts/run_ms3d.sh
"
```

Filter to pedestrian-only pseudo labels:

```bash
docker exec ms3d_training bash -lc "
cd /MS3D/tools
python cfgs/target_custom/label_generation/round1_nusc_pv_rcnn/scripts/filter_pedestrian_ps.py \
  --input /MS3D/tools/cfgs/target_custom/label_generation/round1_nusc_pv_rcnn/ps_labels/final_ps_dict.pkl \
  --output /MS3D/tools/cfgs/target_custom/label_generation/round1_nusc_pv_rcnn/ps_labels/final_pseudo_labels_ped_zfit.pkl
"
```

Filter the pedestrian pseudo labels to 15 m:

```bash
docker exec ms3d_training bash -lc "
cd /MS3D/tools
python cfgs/target_custom/label_generation/round1_nusc_pv_rcnn/scripts/filter_ps_by_distance.py \
  --input /MS3D/tools/cfgs/target_custom/label_generation/round1_nusc_pv_rcnn/ps_labels/final_pseudo_labels_ped_zfit.pkl \
  --output /MS3D/tools/cfgs/target_custom/label_generation/round1_nusc_pv_rcnn/ps_labels/final_pseudo_labels_ped_zfit_15m.pkl \
  --max-distance 15.0
"
```

## Training

The default `SELF_TRAIN.INIT_PS` in `ms3d_custom_round1_pv_rcnn_plusplus_points_nusc.yaml` points to the 15 m pedestrian pseudo-label file above.

Run target-only training:

```bash
docker exec ms3d_training bash -lc "
cd /MS3D/tools
export PYTHONPATH=/MS3D:/MS3D/tracker
python train.py \
  --cfg_file cfgs/target_custom/ms3d_custom_round1_pv_rcnn_plusplus_points_nusc.yaml \
  --pretrained_model /MS3D/pcdet/models/nuscenes_uda_pv_rcnn_plusplus_resnet_anchorhead_10xyzt_allcls.pth \
  --custom_target_scenes \
  --batch_size 4 \
  --epochs 30 \
  --workers 4 \
  --extra_tag nusc_points_v4_pedps_bs4_e30
"
```

## Inference And GT Evaluation

Save raw predictions without GT metrics:

```bash
docker exec ms3d_training bash -lc "
cd /MS3D/tools
export PYTHONPATH=/MS3D:/MS3D/tracker
python test.py \
  --cfg_file cfgs/target_custom/ms3d_custom_round1_pv_rcnn_plusplus_points_nusc.yaml \
  --ckpt /MS3D/output/target_custom/ms3d_custom_round1_pv_rcnn_plusplus_points_nusc/nusc_points_v4_pedps_bs4_e30/ckpt/checkpoint_epoch_30.pth \
  --eval_tag nusc_points_v4_result \
  --batch_size 4 \
  --custom_target_scenes \
  --save_to_file \
  --set \
    MODEL.POST_PROCESSING.EVAL_METRIC none \
    MODEL.POST_PROCESSING.SCORE_THRESH 0.01
"
```

Evaluate against `custom_eval_gt_v1` with KITTI metrics:

```bash
docker exec ms3d_training bash -lc "
cd /MS3D/tools
export PYTHONPATH=/MS3D:/MS3D/tracker
python test.py \
  --cfg_file cfgs/target_custom/ms3d_custom_round1_pv_rcnn_plusplus_points_nusc.yaml \
  --ckpt /MS3D/output/target_custom/ms3d_custom_round1_pv_rcnn_plusplus_points_nusc/nusc_points_v4_pedps_bs4_e30/ckpt/checkpoint_epoch_30.pth \
  --eval_tag nusc_points_v4_metric \
  --target_dataset custom \
  --batch_size 4 \
  --sweeps 1 \
  --use_tta 0 \
  --save_to_file \
  --set \
    DATA_CONFIG_TAR.DATA_PATH /MS3D/data/custom_eval_gt_v1 \
    DATA_CONFIG_TAR.DATA_SPLIT.test val \
    DATA_CONFIG_TAR.INFO_PATH.val \"[custom_infos_val.pkl]\" \
    MODEL.POST_PROCESSING.EVAL_METRIC kitti \
    MODEL.POST_PROCESSING.SCORE_THRESH 0.01
"
```

The custom evaluation path saves PR-curve artifacts next to the eval output.

## Optional Visualization

X11 forwarding is required before interactive Open3D commands:

```bash
xhost +local:root
```

Visualize pseudo labels:

```bash
docker exec -it ms3d_training bash -lc "
cd /MS3D/tools
export PYTHONPATH=/MS3D:/MS3D/tracker
export DISPLAY=:1
python visualize_3d.py \
  --cfg_file cfgs/dataset_configs/custom_points_dataset_da_v2.yaml \
  --ps_pkl /MS3D/tools/cfgs/target_custom/label_generation/round1_nusc_pv_rcnn/ps_labels/final_pseudo_labels_ped_zfit_15m.pkl \
  --split train \
  --idx 0 \
  --ps_score_th 0.01 \
  --interactive_step
"
```

Visualize a trained checkpoint:

```bash
docker exec -it ms3d_training bash -lc "
cd /MS3D/tools
export PYTHONPATH=/MS3D:/MS3D/tracker
export DISPLAY=:1
python visualize_3d.py \
  --cfg_file cfgs/target_custom/ms3d_custom_round1_pv_rcnn_plusplus_points_nusc.yaml \
  --ckpt /MS3D/output/target_custom/ms3d_custom_round1_pv_rcnn_plusplus_points_nusc/nusc_points_v4_pedps_bs4_e30/ckpt/checkpoint_epoch_30.pth \
  --splits train \
  --idx 0 \
  --pred_score_th 0.01 \
  --interactive_step
"
```

## Optional Video Export

Render a video from checkpoint predictions:

```bash
docker exec -it ms3d_training bash -lc "
cd /MS3D/tools
export PYTHONPATH=/MS3D:/MS3D/tracker
export DISPLAY=:1
python make_video_v2.py \
  --cfg_file cfgs/target_custom/ms3d_custom_round1_pv_rcnn_plusplus_points_nusc.yaml \
  --ckpt /MS3D/output/target_custom/ms3d_custom_round1_pv_rcnn_plusplus_points_nusc/nusc_points_v4_pedps_bs4_e30/ckpt/checkpoint_epoch_30.pth \
  --camera_json /MS3D/tools/DepthCamera_2026-03-25-18-20-52.json \
  --splits train \
  --frame_start 0 \
  --frame_end 1200 \
  --fps 10 \
  --line_width 15.0 \
  --output_dir /MS3D/output/video_exports/nusc_points_v4_preview
"
```
