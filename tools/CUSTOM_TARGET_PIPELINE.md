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
docker run --rm --gpus all \
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
docker run --rm --gpus all \
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
docker run --rm --gpus all \
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
