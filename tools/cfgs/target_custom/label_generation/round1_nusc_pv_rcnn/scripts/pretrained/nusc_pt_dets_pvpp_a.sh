#!/bin/bash

python test.py \
  --cfg_file cfgs/nuscenes_models/uda_pv_rcnn_plusplus_resnet_anchorhead.yaml \
  --ckpt /MS3D/pcdet/models/nuscenes_uda_pv_rcnn_plusplus_resnet_anchorhead_10xyzt_allcls.pth \
  --eval_tag nusc_custom_points_trainall \
  --target_dataset custom \
  --sweeps 2 \
  --batch_size 4 \
  --use_tta 0 \
  --set \
    DATA_CONFIG_TAR.DATA_PATH /MS3D/data/custom_points_ps_dataset_v2 \
    DATA_CONFIG_TAR.DATA_SPLIT.test train \
    DATA_CONFIG_TAR.INFO_PATH.test "[custom_infos_train.pkl]" \
    MODEL.POST_PROCESSING.EVAL_METRIC none \
    MODEL.POST_PROCESSING.SCORE_THRESH 0.01