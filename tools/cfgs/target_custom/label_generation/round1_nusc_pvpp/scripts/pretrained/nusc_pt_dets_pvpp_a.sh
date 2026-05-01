#!/bin/bash

python test.py \
    --cfg_file cfgs/nuscenes_models/uda_pv_rcnn_plusplus_resnet_anchorhead.yaml \
    --ckpt /MS3D/pcdet/models/nuscenes_uda_pv_rcnn_plusplus_resnet_anchorhead_10xyzt_allcls.pth \
    --eval_tag nusc10xyzt_custom1xyzt_notta_pvpp_anchor \
    --target_dataset custom --sweeps 1 --batch_size 4 --use_tta 0 \
    --set DATA_CONFIG_TAR.DATA_SPLIT.test train MODEL.POST_PROCESSING.EVAL_METRIC none MODEL.POST_PROCESSING.SCORE_THRESH 0.01
