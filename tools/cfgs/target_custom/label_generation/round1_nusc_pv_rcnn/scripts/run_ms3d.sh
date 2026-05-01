#!/bin/bash

export PYTHONPATH="/MS3D:/MS3D/tracker:${PYTHONPATH}"

python ensemble_kbf.py --ps_cfg /MS3D/tools/cfgs/target_custom/label_generation/round1_nusc_pv_rcnn/cfgs/ps_config.yaml
python generate_tracks.py --ps_cfg /MS3D/tools/cfgs/target_custom/label_generation/round1_nusc_pv_rcnn/cfgs/ps_config.yaml --cls_id 1
python generate_tracks.py --ps_cfg /MS3D/tools/cfgs/target_custom/label_generation/round1_nusc_pv_rcnn/cfgs/ps_config.yaml --cls_id 1 --static_veh
python generate_tracks.py --ps_cfg /MS3D/tools/cfgs/target_custom/label_generation/round1_nusc_pv_rcnn/cfgs/ps_config.yaml --cls_id 2
python temporal_refinement.py --ps_cfg /MS3D/tools/cfgs/target_custom/label_generation/round1_nusc_pv_rcnn/cfgs/ps_config.yaml
