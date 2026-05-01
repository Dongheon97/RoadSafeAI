'''
# visualize pickle files:
xhost +local:root
docker exec -it ms3d_training bash -lc "
cd /MS3D/tools
export PYTHONPATH=/MS3D:/MS3D/tracker
export DISPLAY=:1
python visualize_3d.py \
  --cfg_file cfgs/dataset_configs/custom_points_dataset_da_v2.yaml \
  --ps_pkl /MS3D/tools/cfgs/target_custom/label_generation/custom_points_round1/ps_labels/final_pseudo_labels_ped_zfit_15m.pkl \
  --split train \
  --idx 0 \
  --ps_score_th 0.005 \
  --interactive_step
"
docker exec ms3d_training bash -lc '
cd /MS3D/tools
export PYTHONPATH=/MS3D:/MS3D/tracker
export DISPLAY=:1            
python visualize_3d.py \
  --cfg_file /MS3D/tools/cfgs/target_custom/ms3d_custom_round1_pv_rcnn_plusplus_points_nusc.yaml \
  --ckpt /MS3D/output/target_custom/ms3d_custom_round1_pv_rcnn_plusplus_points_nusc/nusc_points_v3_pedps_bs4_e30/ckpt/checkpoint_epoch_30.pth \
  --splits train \
  --idx 0 \
  --pred_score_th 0.01 \
  --interactive_step 
'
'''
import torch
from pathlib import Path
import sys
sys.path.append('/MS3D')
from pcdet.models import build_network, load_data_to_gpu
import copy
from pcdet.utils import common_utils
from pcdet.datasets import build_dataloader
import open3d as o3d
from visual_utils import open3d_vis_utils as V
import argparse
import pickle
import os
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import box_fusion_utils
from pcdet.utils import compatibility_utils as compat
import numpy as np
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMERA_SAVE_PATH = REPO_ROOT / 'output' / 'visualize_3d' / 'last_camera.json'
DEFAULT_VIEW_BY_MODE = {
    'det_pkl': {
        'front': [-0.009079385782427972, -0.79382993606647601, 0.60807203303433444],
        'lookat': [0.13592805125144847, 25.565951040207825, -13.855443454771956],
        'up': [-0.008889802784222859, 0.60813714531420293, 0.79378220180068892],
        'zoom': 0.21999999999999992,
    },
    'ps_pkl': {
        'front': [-0.009079385782427972, -0.79382993606647601, 0.60807203303433444],
        'lookat': [0.13592805125144847, 25.565951040207825, -13.855443454771956],
        'up': [-0.008889802784222859, 0.60813714531420293, 0.79378220180068892],
        'zoom': 0.21999999999999992,
    },
    'dets_txt': {
        'front': [0.72737973442893356, -0.51797808311597837, 0.45013045592760198],
        'lookat': [-13.773417658854088, 0.062465858514556709, -0.53706070047660459],
        'up': [-0.37595030931731882, 0.2479623453125949, 0.89284715390221758],
        'zoom': 0.079999999999999946,
    },
    'ckpt': {
        'front': [-0.009079385782427972, -0.79382993606647601, 0.60807203303433444],
        'lookat': [0.13592805125144847, 25.565951040207825, -13.855443454771956],
        'up': [-0.008889802784222859, 0.60813714531420293, 0.79378220180068892],
        'zoom': 0.21999999999999992,
    },
}
"""
# Examples
python visualize_3d.py --cfg_file cfgs/target-nuscenes/ft_waymo_secondiou.yaml  \
    --idx 6 --dets_txt cfgs/target-nuscenes/raw_dets/det_1f_paths.txt

python visualize_3d.py --cfg_file cfgs/target-nuscenes/ft_waymo_secondiou.yaml \
    --ps_pkl ../output/target-nuscenes/ft_waymo_secondiou/default/ps_label/ps_label_e0.pkl \
    --split train --custom_train_split --idx 6

"""

def main():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='/MS3D/tools/cfgs/dataset_configs/waymo_dataset_da.yaml',
                        help='just use the target dataset cfg file')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='specify the model ckpt path')    
    parser.add_argument('--det_pkl', type=str, required=False,
                        help='These are the result.pkl files from test.py')
    parser.add_argument('--ps_pkl', type=str, required=False,
                        help='These are the ps_dict_*, ps_label_e*.pkl files generated from MS3D')
    parser.add_argument('--ps_pkl2', type=str, required=False, default=None,
                        help='These are the ps_dict_*, ps_label_e*.pkl files generated from MS3D')
    parser.add_argument('--dets_txt', type=str, default=None, required=False,
                        help='det_*f_paths.txt file containing detector pkl paths')                        
    parser.add_argument('--idx', type=int, default=0,
                        help='If you wish to only display a certain frame index')
    parser.add_argument('--split', type=str, default='train',
                        help='Specify train or test split')    
    parser.add_argument('--sampled_interval', type=int, default=None,
                        help='same as SAMPLED_INTERVAL config parameter')        
    parser.add_argument('--save_video_dir', type=str, required=False, default='save_frames')
    parser.add_argument('--custom_train_split', action='store_true', default=False)
    parser.add_argument('--save_video', action='store_true', default=False)
    parser.add_argument('--show_gt', action='store_true', default=False)
    parser.add_argument('--sweeps', type=int, default=None)
    parser.add_argument('--bev_vis', action='store_true', default=False)
    parser.add_argument('--use_linemesh', action='store_true', default=False, help='Visualize with larger boxes but very slow render time')
    parser.add_argument('--use_class_colors', action='store_true', default=False)    
    parser.add_argument('--pointcloud_only', action='store_true', default=False)    
    parser.add_argument('--interactive_step', action='store_true', default=False,
                        help='Interactive detection viewer. Press Space for next frame, Q to quit.')
    parser.add_argument('--pred_score_th', type=float, default=0.05,
                        help='Prediction score threshold for displaying boxes in interactive_step mode.')
    parser.add_argument('--ps_score_th', type=float, default=0.4,
                        help='Pseudo-label score threshold for displaying boxes from --ps_pkl.')
    parser.add_argument('--pkl_points_mode', type=str, default='raw_1frame',
                        choices=['raw_1frame', 'dataset_processed'],
                        help='Point source for --det_pkl/--ps_pkl visualization. '
                             'raw_1frame bypasses dataset preprocessing and SHIFT_COOR for easier '
                             'coordinate debugging; dataset_processed preserves the previous behavior.')
    parser.add_argument('--theme', type=str, default='dark', choices=['dark', 'light'],
                        help='Open3D viewer theme.')
    parser.add_argument('--point_color_mode', type=str, default='height', choices=['height', 'mono'],
                        help='Point cloud coloring mode.')
    parser.add_argument('--point_size', type=float, default=2.0,
                        help='Open3D point size for all viewer modes.')
    parser.add_argument('--line_width', type=float, default=4.0,
                        help='Open3D line width for non-linemesh bounding boxes.')
    parser.add_argument('--window_x', type=int, default=2000,
                        help='Open3D window left position in pixels.')
    parser.add_argument('--window_y', type=int, default=300,
                        help='Open3D window top position in pixels.')
    parser.add_argument('--window_width', type=int, default=4000,
                        help='Open3D window width in pixels.')
    parser.add_argument('--window_height', type=int, default=3000,
                        help='Open3D window height in pixels.')
    parser.add_argument('--camera_json', type=str, default=None,
                        help='Optional camera json to load at startup. Relative paths are resolved from repo root.')
    parser.add_argument('--camera_save_path', type=str, default=str(DEFAULT_CAMERA_SAVE_PATH),
                        help='Path used for saving and auto-loading the camera viewpoint.')
    args = parser.parse_args()

    def apply_theme(vis_obj):
        V.apply_render_theme(
            vis_obj,
            theme=args.theme,
            point_size=args.point_size,
            line_width=args.line_width,
        )

    def resolve_repo_path(path_str):
        path = Path(path_str)
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
        return path

    def create_window_or_raise(vis_obj, window_name):
        window_ok = vis_obj.create_window(
            window_name=window_name,
            left=args.window_x,
            top=args.window_y,
            width=args.window_width,
            height=args.window_height
        )
        if not window_ok:
            display = os.environ.get('DISPLAY', '')
            raise RuntimeError(
                'Open3D window creation failed. '
                f'DISPLAY="{display}". '
                'Run container with X11 forwarding (DISPLAY + /tmp/.X11-unix mount).'
            )

    camera_save_path = resolve_repo_path(args.camera_save_path)

    def load_saved_camera_if_exists(path, required=False):
        if not path.exists():
            if required:
                raise FileNotFoundError(f'Camera file not found: {path}')
            return None
        try:
            return o3d.io.read_pinhole_camera_parameters(str(path))
        except Exception as exc:
            if required:
                raise RuntimeError(f'Failed to load camera parameters from {path}: {exc}') from exc
            print(f'Warning: failed to load camera parameters from {path}: {exc}. Falling back to default view.')
            return None

    camera_state = {'params': None}
    if args.camera_json is not None:
        explicit_camera_path = resolve_repo_path(args.camera_json)
        camera_state['params'] = load_saved_camera_if_exists(explicit_camera_path, required=True)
        print(f'Loaded camera parameters from {explicit_camera_path}')
    else:
        camera_state['params'] = load_saved_camera_if_exists(camera_save_path, required=False)
        if camera_state['params'] is not None:
            print(f'Loaded saved camera parameters from {camera_save_path}')

    def apply_default_view(vis_obj, mode_name):
        ctr = vis_obj.get_view_control()
        if ctr is None:
            return
        view = DEFAULT_VIEW_BY_MODE[mode_name]
        ctr.set_front(view['front'])
        ctr.set_lookat(view['lookat'])
        ctr.set_up(view['up'])
        ctr.set_zoom(view['zoom'])

    def apply_saved_camera(vis_obj):
        if camera_state['params'] is None:
            return False
        ctr = vis_obj.get_view_control()
        if ctr is None:
            return False
        try:
            try:
                applied = ctr.convert_from_pinhole_camera_parameters(camera_state['params'], allow_arbitrary=True)
            except TypeError:
                applied = ctr.convert_from_pinhole_camera_parameters(camera_state['params'])
        except Exception as exc:
            print(f'Warning: failed to apply saved camera: {exc}. Falling back to default view.')
            return False
        if applied is False:
            print('Warning: saved camera was rejected by Open3D. Falling back to default view.')
            return False
        return True

    def apply_view(vis_obj, mode_name):
        if not apply_saved_camera(vis_obj):
            apply_default_view(vis_obj, mode_name)

    def save_current_camera(vis_obj):
        ctr = vis_obj.get_view_control()
        if ctr is None:
            print('Camera save skipped: Open3D view control is unavailable.')
            return False
        try:
            camera_params = ctr.convert_to_pinhole_camera_parameters()
            camera_save_path.parent.mkdir(parents=True, exist_ok=True)
            ok = o3d.io.write_pinhole_camera_parameters(str(camera_save_path), camera_params)
        except Exception as exc:
            print(f'Failed to save camera parameters to {camera_save_path}: {exc}')
            return False
        if not ok:
            print(f'Failed to save camera parameters to {camera_save_path}')
            return False
        camera_state['params'] = camera_params
        print(f'Saved camera parameters to {camera_save_path}. This viewpoint will be reused for subsequent frames.')
        return False

    geom_kwargs = {
        'use_linemesh': args.use_linemesh,
        'use_class_colors': args.use_class_colors,
        'theme': args.theme,
        'point_color_mode': args.point_color_mode,
    }
    
    if args.bev_vis:
        from visualize_bev import plot_boxes
        import matplotlib.pyplot as plt

    # Load dataset or model+dataset
    cfg_from_yaml_file(args.cfg_file, cfg)
    if 'dataset_configs' in args.cfg_file:
        data_config = cfg      
    else:
        if cfg.get('DATA_CONFIG_TAR', None):
            data_config = cfg.DATA_CONFIG_TAR
        else:
            data_config = cfg.DATA_CONFIG

    cls_names = data_config.CLASS_NAMES
    data_config.DATA_SPLIT.test = args.split
    data_config.USE_TTA = False
    if args.sampled_interval is not None:          
        data_config.SAMPLED_INTERVAL.test = args.sampled_interval
    data_config.USE_CUSTOM_TRAIN_SCENES = args.custom_train_split
    logger = common_utils.create_logger('temp.txt', rank=cfg.LOCAL_RANK)

    if data_config.get('MAX_SWEEPS',False):
        if args.sweeps is not None:
            data_config.MAX_SWEEPS = args.sweeps

    if data_config.get('SEQUENCE_CONFIG',False):
        if data_config.SEQUENCE_CONFIG.ENABLED:
            if args.sweeps is not None:
                data_config.SEQUENCE_CONFIG.SAMPLE_OFFSET = [-(args.sweeps-1),0]

            # data_config.POINT_FEATURE_ENCODING.src_feature_list=['x','y','z','intensity','timestamp']
            data_config.POINT_FEATURE_ENCODING.src_feature_list=['x', 'y', 'z', 'intensity', 'elongation', 'timestamp']
            data_config.POINT_FEATURE_ENCODING.used_feature_list=['x','y','z','timestamp']        

    target_set, target_loader, _ = build_dataloader(
            dataset_cfg=data_config,
            class_names=cls_names,
            batch_size=1, logger=logger, training=False, dist=False, workers=1
        )          

    idx_to_frameid = {v: k for k, v in target_set.frameid_to_idx.items()}

    raw_gt_warning_shown = {'shown': False}

    def get_pkl_render_sample(frame_id):
        ds_idx = target_set.frameid_to_idx[frame_id]
        if args.pkl_points_mode == 'dataset_processed':
            sample = target_set[ds_idx]
            return sample['points'], sample['gt_boxes']

        raw_points = np.asarray(compat.get_lidar(target_set, frame_id))
        if raw_points.ndim != 2 or raw_points.shape[1] < 3:
            raise ValueError(
                f'Expected raw lidar with shape (N, >=3) for frame {frame_id}, '
                f'got {raw_points.shape}'
            )

        if args.show_gt and not raw_gt_warning_shown['shown']:
            print('Warning: --show_gt is disabled in pkl raw_1frame mode because GT boxes are '
                  'not remapped into the raw point frame.')
            raw_gt_warning_shown['shown'] = True

        return raw_points, None

    # If no pkl file, just show point cloud and gt boxes (optional)
    if (args.det_pkl is None) and (args.ps_pkl is None) and (args.dets_txt is None) and (args.ckpt is None):    
        for idx, data_dict in enumerate(target_loader):
            if idx < args.idx:
                print(f'Skipping {idx}/{args.idx}')
                continue
            V.draw_scenes(points=data_dict['points'][:, 1:], 
                          gt_boxes=data_dict['gt_boxes'][0] if args.show_gt else None,                           
                          draw_origin=False, use_linemesh=args.use_linemesh,
                          ref_labels=list(data_dict['gt_boxes'][0][:,7].astype(int)),
                          window_x=args.window_x, window_y=args.window_y,
                          window_width=args.window_width, window_height=args.window_height,
                          theme=args.theme, point_color_mode=args.point_color_mode,
                          point_size=args.point_size, line_width=args.line_width)

    # Visualize pkls
    if (args.det_pkl is not None) or (args.ps_pkl is not None) or (args.dets_txt is not None):    

        # Load detection pickle
        if args.det_pkl is not None:
            with open(args.det_pkl,'rb') as f:
                det_annos = pickle.load(f)
            valid_indices = [
                idx for idx, det_anno in enumerate(det_annos)
                if det_anno['frame_id'] in target_set.frameid_to_idx.keys()
            ]
            if not valid_indices:
                print('No valid frames found in det_pkl for current dataset split.')
                return

            start_pos = min(max(args.idx, 0), len(valid_indices) - 1)
            state = {'pos': start_pos}

            vis = o3d.visualization.VisualizerWithKeyCallback()
            create_window_or_raise(vis, 'RoadSafeAI PKL Step Viewer')

            def render_det_frame(vis_obj):
                det_idx = valid_indices[state['pos']]
                det_anno = det_annos[det_idx]
                frame_id = det_anno['frame_id']
                pts, gt_boxes = get_pkl_render_sample(frame_id)
                score_mask = det_anno['score'] > args.pred_score_th
                geom = V.get_geometries(
                    points=pts,
                    gt_boxes=gt_boxes if args.show_gt else None,
                    ref_boxes=det_anno['boxes_lidar'][score_mask],
                    ref_scores=det_anno['score'][score_mask],
                    ref_labels=[1 for _ in range(int(score_mask.sum()))],
                    **geom_kwargs
                )
                vis_obj.clear_geometries()
                for g in geom:
                    vis_obj.add_geometry(g)
                apply_view(vis_obj, 'det_pkl')
                apply_theme(vis_obj)
                vis_obj.update_renderer()
                print(
                    f"[det_pkl {state['pos']+1}/{len(valid_indices)}] "
                    f"frame_idx={det_idx} frame_id={frame_id} shown_boxes={int(score_mask.sum())}"
                )

            def on_space(vis_obj):
                if state['pos'] >= len(valid_indices) - 1:
                    print('Reached last frame.')
                    return False
                state['pos'] += 1
                render_det_frame(vis_obj)
                return False

            def on_prev(vis_obj):
                if state['pos'] <= 0:
                    print('Reached first frame.')
                    return False
                state['pos'] -= 1
                render_det_frame(vis_obj)
                return False

            def on_q(vis_obj):
                vis_obj.close()
                return False

            vis.register_key_callback(ord(' '), on_space)
            vis.register_key_callback(ord('N'), on_space)
            vis.register_key_callback(262, on_space)
            vis.register_key_callback(ord('B'), on_prev)
            vis.register_key_callback(263, on_prev)
            vis.register_key_callback(ord('Q'), on_q)
            vis.register_key_callback(256, on_q)
            vis.register_key_callback(ord('S'), save_current_camera)
            print('PKL mode: Space/N/Right = next, B/Left = previous, S = save camera, Q/Esc = quit')
            render_det_frame(vis)
            vis.run()
            vis.destroy_window()
            return
        if args.ps_pkl is not None:
            with open(args.ps_pkl,'rb') as f:
                ps_dict = pickle.load(f)

            if args.ps_pkl2 is not None:
                with open(args.ps_pkl2,'rb') as f:
                    ps_dict2 = pickle.load(f)
            valid_indices = [
                idx for idx in range(len(target_set))
                if idx_to_frameid[idx] in ps_dict.keys()
            ]
            if not valid_indices:
                print('No valid frames found in ps_pkl for current dataset split.')
                return

            start_pos = min(max(args.idx, 0), len(valid_indices) - 1)
            state = {'pos': start_pos}

            vis = o3d.visualization.VisualizerWithKeyCallback()
            create_window_or_raise(vis, 'RoadSafeAI PKL Step Viewer')

            def render_ps_frame(vis_obj):
                ds_idx = valid_indices[state['pos']]
                frame_id = idx_to_frameid[ds_idx]
                pts, gt_boxes = get_pkl_render_sample(frame_id)
                ps_mask = ps_dict[frame_id]['gt_boxes'][:,8] > args.ps_score_th
                ref_boxes2 = ps_dict2[frame_id]['gt_boxes'][:,:7] if args.ps_pkl2 is not None else None
                geom = V.get_geometries(
                    points=pts,
                    gt_boxes=gt_boxes if args.show_gt else None,
                    ref_boxes=ps_dict[frame_id]['gt_boxes'][:,:7][ps_mask],
                    ref_boxes2=ref_boxes2,
                    ref_scores=ps_dict[frame_id]['gt_boxes'][:,8][ps_mask],
                    ref_labels=list(abs(ps_dict[frame_id]['gt_boxes'][:,7][ps_mask].astype(int))),
                    **geom_kwargs
                )
                vis_obj.clear_geometries()
                for g in geom:
                    vis_obj.add_geometry(g)
                apply_view(vis_obj, 'ps_pkl')
                apply_theme(vis_obj)
                vis_obj.update_renderer()
                shown_boxes = int(ps_mask.sum())
                print(
                    f"[ps_pkl {state['pos']+1}/{len(valid_indices)}] "
                    f"dataset_idx={ds_idx} frame_id={frame_id} shown_boxes={shown_boxes} "
                    f"ps_score_th={args.ps_score_th:.2f}"
                )

            def on_space(vis_obj):
                if state['pos'] >= len(valid_indices) - 1:
                    print('Reached last frame.')
                    return False
                state['pos'] += 1
                render_ps_frame(vis_obj)
                return False

            def on_prev(vis_obj):
                if state['pos'] <= 0:
                    print('Reached first frame.')
                    return False
                state['pos'] -= 1
                render_ps_frame(vis_obj)
                return False

            def on_q(vis_obj):
                vis_obj.close()
                return False

            vis.register_key_callback(ord(' '), on_space)
            vis.register_key_callback(ord('N'), on_space)
            vis.register_key_callback(262, on_space)
            vis.register_key_callback(ord('B'), on_prev)
            vis.register_key_callback(263, on_prev)
            vis.register_key_callback(ord('Q'), on_q)
            vis.register_key_callback(256, on_q)
            vis.register_key_callback(ord('S'), save_current_camera)
            print('PKL mode: Space/N/Right = next, B/Left = previous, S = save camera, Q/Esc = quit')
            render_ps_frame(vis)
            vis.run()
            vis.destroy_window()
            return
        else:                        
            det_annos = box_fusion_utils.load_src_paths_txt(args.dets_txt)
            src_keys = list(det_annos.keys())
            src_keys.remove('det_cls_weights')
            len_data = len(det_annos[src_keys[0]])
            valid_indices = [
                idx for idx in range(len_data)
                if det_annos[src_keys[0]][idx]['frame_id'] in target_set.frameid_to_idx.keys()
            ]
            if not valid_indices:
                print('No valid frames found in dets_txt for current dataset split.')
                return

            start_pos = min(max(args.idx, 0), len(valid_indices) - 1)
            state = {'pos': start_pos}

            vis = o3d.visualization.VisualizerWithKeyCallback()
            create_window_or_raise(vis, 'RoadSafeAI PKL Step Viewer')

            cmap = np.array([[49,131,106],[193, 107, 107],[110, 163, 167],[214, 206, 114],[49,131,106],[110, 163, 167],[214, 206, 114]])/255

            def render_msda_frame(vis_obj):
                det_idx = valid_indices[state['pos']]
                frame_id = det_annos[src_keys[0]][det_idx]['frame_id']
                sample = target_set[target_set.frameid_to_idx[frame_id]]
                geom = []
                for sid, key in enumerate(src_keys):
                    points = sample['points'] if sid == 0 else None
                    mask = det_annos[key][det_idx]['score'] > 0.2
                    geom.extend(V.get_geometries(
                        points=points,
                        ref_boxes=det_annos[key][det_idx]['boxes_lidar'][mask],
                        ref_scores=det_annos[key][det_idx]['score'][mask],
                        ref_labels=[1 for _ in range(int(mask.sum()))],
                        ref_box_colors=cmap[sid % len(cmap)],
                        gt_boxes=sample['gt_boxes'] if args.show_gt else None,
                        draw_origin=False,
                        line_thickness=0.055,
                        use_linemesh=args.use_linemesh,
                        use_class_colors=args.use_class_colors,
                        theme=args.theme,
                        point_color_mode=args.point_color_mode
                    ))
                vis_obj.clear_geometries()
                for g in geom:
                    vis_obj.add_geometry(g)
                apply_view(vis_obj, 'dets_txt')
                apply_theme(vis_obj)
                vis_obj.update_renderer()
                print(f"[dets_txt {state['pos']+1}/{len(valid_indices)}] frame_idx={det_idx} frame_id={frame_id}")

            def on_space(vis_obj):
                if state['pos'] >= len(valid_indices) - 1:
                    print('Reached last frame.')
                    return False
                state['pos'] += 1
                render_msda_frame(vis_obj)
                return False

            def on_prev(vis_obj):
                if state['pos'] <= 0:
                    print('Reached first frame.')
                    return False
                state['pos'] -= 1
                render_msda_frame(vis_obj)
                return False

            def on_q(vis_obj):
                vis_obj.close()
                return False

            vis.register_key_callback(ord(' '), on_space)
            vis.register_key_callback(ord('N'), on_space)
            vis.register_key_callback(262, on_space)
            vis.register_key_callback(ord('B'), on_prev)
            vis.register_key_callback(263, on_prev)
            vis.register_key_callback(ord('Q'), on_q)
            vis.register_key_callback(256, on_q)
            vis.register_key_callback(ord('S'), save_current_camera)
            print('PKL mode: Space/N/Right = next, B/Left = previous, S = save camera, Q/Esc = quit')
            render_msda_frame(vis)
            vis.run()
            vis.destroy_window()
            return
            
    else:
        # Load trained model for inference
        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=target_set)
        model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
        model.cuda()
        model.eval()

        if args.interactive_step:
            start_idx = max(0, min(args.idx, len(target_set) - 1))
            state = {'idx': start_idx}
            model_class_names = list(cfg.CLASS_NAMES) if hasattr(cfg, 'CLASS_NAMES') else list(cls_names)

            def summarize_class_counts(label_tensor):
                if label_tensor is None or label_tensor.numel() == 0:
                    return 'none'
                labels = label_tensor.detach().cpu().numpy().astype(np.int32)
                class_counts = {}
                for label_id in labels:
                    class_idx = int(label_id) - 1
                    if 0 <= class_idx < len(model_class_names):
                        class_name = model_class_names[class_idx]
                    else:
                        class_name = f'id_{int(label_id)}'
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                return ', '.join([f'{k}:{v}' for k, v in sorted(class_counts.items())])

            vis = o3d.visualization.VisualizerWithKeyCallback()
            create_window_or_raise(vis, 'RoadSafeAI Detection Step Viewer')

            def render_frame(vis_obj):
                data_dict = target_set.collate_batch([target_set[state['idx']]])
                load_data_to_gpu(data_dict)

                with torch.no_grad():
                    pred_dicts, _ = model.forward(data_dict)

                pred_boxes = pred_dicts[0]['pred_boxes']
                pred_labels = pred_dicts[0]['pred_labels']
                pred_scores = pred_dicts[0]['pred_scores']

                score_mask = pred_scores >= args.pred_score_th
                shown_boxes = pred_boxes[score_mask]
                shown_labels = pred_labels[score_mask]
                shown_scores = pred_scores[score_mask]

                gt_boxes = data_dict['gt_boxes'][0] if args.show_gt and 'gt_boxes' in data_dict else None
                geom = V.get_geometries(
                    points=data_dict['points'][:, 1:],
                    gt_boxes=gt_boxes,
                    ref_boxes=shown_boxes,
                    ref_scores=shown_scores,
                    ref_labels=shown_labels,
                    **geom_kwargs
                )

                vis_obj.clear_geometries()
                for g in geom:
                    vis_obj.add_geometry(g)

                apply_view(vis_obj, 'ckpt')
                apply_theme(vis_obj)
                vis_obj.update_renderer()

                frame_id = data_dict['frame_id'][0] if 'frame_id' in data_dict else str(state['idx'])
                total_boxes = int(pred_scores.shape[0])
                shown_count = int(score_mask.sum().item())
                total_classes = summarize_class_counts(pred_labels)
                shown_classes = summarize_class_counts(shown_labels)
                print(
                    f"[idx {state['idx']:04d}] frame_id={frame_id} "
                    f"pred_total={total_boxes} shown(>={args.pred_score_th:.2f})={shown_count} "
                    f"bbox_exists={shown_count > 0} "
                    f"all_classes=[{total_classes}] shown_classes=[{shown_classes}]"
                )

            def on_space(vis_obj):
                if state['idx'] >= len(target_set) - 1:
                    print('Reached last frame.')
                    return False
                state['idx'] += 1
                render_frame(vis_obj)
                return False

            def on_q(vis_obj):
                vis_obj.close()
                return False

            vis.register_key_callback(ord(' '), on_space)
            vis.register_key_callback(ord('Q'), on_q)
            vis.register_key_callback(ord('N'), on_space)
            vis.register_key_callback(ord('S'), save_current_camera)

            print('Interactive mode: Space/N = next frame, S = save camera, Q = quit')
            print(f"Starting from idx={state['idx']}, split={args.split}, score_th={args.pred_score_th:.2f}")
            render_frame(vis)
            vis.run()
            vis.destroy_window()
            return

        if args.save_video:
            vis = o3d.visualization.Visualizer()
            vis.create_window(
                left=args.window_x,
                top=args.window_y,
                width=args.window_width,
                height=args.window_height
            )
            apply_theme(vis)
        
        with torch.no_grad():
            for idx, data_dict in enumerate(target_loader):
                if idx < args.idx:
                    print(f'Skipping {idx}/{args.idx}')
                    continue
                
                load_data_to_gpu(data_dict)
                print(f'\nVisualizing frame: {idx}')  
                print('Points: ', data_dict['points'].shape[0])

                pred_dicts, _ = model.forward(data_dict)   
                if 'gt_boxes' in data_dict.keys():
                    gt_boxes = data_dict['gt_boxes'][0]

                    # For filtering out gt boxes with 0 pts in waymo scenes
                    # class_mask = np.in1d(target_set.infos[idx]['annos']['name'], target_set.class_names)
                    # class_num_pts = target_set.infos[idx]['annos']['num_points_in_gt'][class_mask]
                    # gt_boxes = gt_boxes[class_num_pts > 0]
                else:
                    gt_boxes = None
   
                if args.save_video:
                    
                    
                    if args.pointcloud_only:
                        geom = V.get_geometries(
                                    points=data_dict['points'][:, 1:],
                                    theme=args.theme,
                                    point_color_mode=args.point_color_mode)
                    else:
                        geom = V.get_geometries(
                                points=data_dict['points'][:, 1:], gt_boxes=gt_boxes if args.show_gt else None, ref_boxes=pred_dicts[0]['pred_boxes'], 
                                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels'],
                                use_linemesh=args.use_linemesh, use_class_colors=args.use_class_colors,
                                theme=args.theme, point_color_mode=args.point_color_mode
                            )
                        
                    vis.clear_geometries()
                    for g in geom:                
                        vis.add_geometry(g)
                        
                    ctr = vis.get_view_control()    
                    # vis.get_render_option().point_size = 1.0

                    # ctr.set_front([ -0.63703010546300987, 0.031621802576535914, 0.77019004559627835 ])
                    # ctr.set_lookat([ 24.513087638782164, 2.8039754478129324, -1.0959739482593087 ])
                    # ctr.set_up([ 0.77082292993229895, 0.019695916704257327, 0.63674491089899199 ])
                    # ctr.set_zoom(0.16)

                    # MS3D++ tgt_waymo qualitative
                    ctr.set_front([ -0.79490788243448329, 0.01495959113947927, 0.60654568589387037 ])
                    ctr.set_lookat([ 15.417785290867977, -1.6179187751048014, -8.5173845851153143 ])
                    ctr.set_up([0.60625059182523544, -0.020154889264250107, 0.79501823900480273])
                    ctr.set_zoom(0.17999999999999994)
                    apply_theme(vis)

                    # MS3D++ tgt_lyft qualitative
                    # ctr.set_front([  0.79570141514638959, -0.092133463771410615, 0.59864069589989899 ])
                    # ctr.set_lookat([ -12.990834711399467, 2.6371120128719636, -9.8107556449013416 ])
                    # ctr.set_up([-0.59940629219165342, 0.022207236266390835, 0.80013682301120415])
                    # ctr.set_zoom(0.17999999999999994)
                    # vis.get_render_option().point_size = 2.0          

                    # MS3D++ tgt_nusc qualitative
                    # ctr.set_front([ 0.0087264629048255243, -0.75327240405054774, 0.65765076913288811 ])
                    # ctr.set_lookat([ -1.8461993414538382, 24.009932413395173, -19.054164307594132 ])
                    # ctr.set_up([0.012925990806422497, 0.65770583609822197, 0.75316396085049842])
                    # ctr.set_zoom(0.2599999999999999)
                    # vis.get_render_option().point_size = 2.0       
                    # vis.update_renderer()         
                    # vis.poll_events()

                    # MS3D++ tgt_nusc qualitative (retroactivelabel)
                    # ctr.set_front([ -0.83664444730658971, -0.1199288729786602, 0.53445592354947258 ])
                    # ctr.set_lookat([ 29.967540865562142, 4.7417471161548566, -16.906388849566994 ])
                    # ctr.set_up([0.53322341149418129, 0.044870204273302128, 0.84478367538854582 ])
                    # ctr.set_zoom(0.15999999999999994)
                    # vis.get_render_option().point_size = 4.0    
                    vis.update_renderer()         
                    vis.poll_events()

                    Path(f'demo_data/{args.save_video_dir}').mkdir(parents=True, exist_ok=True)
                    vis.capture_screen_image(f'demo_data/{args.save_video_dir}/frame-{idx}.jpg', do_render=True)

                else:

                    ref_boxes = pred_dicts[0]['pred_boxes']
                    ref_labels = pred_dicts[0]['pred_labels']
                    ref_scores = pred_dicts[0]['pred_scores']

                    print('Frame ID: ', data_dict['frame_id'])
                    print('Predicted: ', int(ref_boxes.shape[0]))
                    print('Ground truth: ', int(gt_boxes.shape[0]))
                    if args.bev_vis:

                        pts = data_dict['points'][:, 1:].cpu().numpy()
                        fig = plt.figure(figsize=(20,20))
                        ax = plt.subplot(111)
                        fig.subplots_adjust(right=0.7)
                        if (args.sweeps is not None) and (args.sweeps > 1):
                            ax.scatter(pts[:,0],pts[:,1],s=0.1, c='black', marker='o')
                        else:
                            ax.scatter(pts[:,0],pts[:,1],s=0.5, c='black', marker='o')

                        plot_boxes(ax, ref_boxes.cpu().numpy(), 
                            scores=ref_scores.cpu().numpy(),
                            source_labels=ref_labels.cpu().numpy(),
                            limit_range=[-80, -80, -5.0, 80, 80, 3.0], color=[0,1,0])
                        
                        if args.show_gt:
                            plot_boxes(ax, gt_boxes.cpu().numpy(), 
                                        scores=None,
                                        source_labels=None,
                                        limit_range=[-80, -80, -5.0, 80, 80, 3.0], color=[0,0,1])

                        ax.set_aspect('equal')
                        plt.show(block=True)
                    else:
                        V.draw_scenes(
                            points=data_dict['points'][:, 1:], gt_boxes=gt_boxes if args.show_gt else None, ref_boxes=ref_boxes, 
                            ref_scores=ref_scores, ref_labels=ref_labels, use_linemesh=args.use_linemesh,
                            use_class_colors=args.use_class_colors,
                            window_x=args.window_x, window_y=args.window_y,
                            window_width=args.window_width, window_height=args.window_height,
                            theme=args.theme, point_color_mode=args.point_color_mode,
                            point_size=args.point_size, line_width=args.line_width
                        )


if __name__ == '__main__':
    main()
