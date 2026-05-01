#!/bin/bash

python test.py \
    --cfg_file cfgs/lyft_models/uda_pv_rcnn_plusplus_resnet_anchorhead.yaml \
    --ckpt /MS3D/pcdet/models/lyft_uda_pv_rcnn_plusplus_resnet_anchorhead_1xyz_allcls.pth \
    --eval_tag lyft1xyzt_custom1xyzt_notta_pvpp_anchor \
    --target_dataset custom --sweeps 1 --batch_size 4 --use_tta 0 \
    --set DATA_CONFIG_TAR.DATA_SPLIT.test train MODEL.POST_PROCESSING.EVAL_METRIC none
