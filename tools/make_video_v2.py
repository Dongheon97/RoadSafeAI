#!/usr/bin/env python3

'''
# Make video from checkpoint predictions:
docker exec -it ms3d_training bash -lc "
cd /MS3D/tools
export PYTHONPATH=/MS3D:/MS3D/tracker
export DISPLAY=:1

python make_video_v2.py \
  --cfg_file /MS3D/tools/cfgs/target_custom/ms3d_custom_round1_pv_rcnn_plusplus_points_nusc.yaml \
  --ckpt /MS3D/output/target_custom/ms3d_custom_round1_pv_rcnn_plusplus_points_nusc/nusc_points_v3_pedps_bs4_e30/ckpt/checkpoint_epoch_30.pth \
  --camera_json /MS3D/output/visualize_3d/side_cam.json \
  --splits train \
  --frame_start 0 \
  --frame_end 1200 \
  --fps 10 \
  --pred_box_color 1,0,0 \
  --line_width 15.0 \
  --output_dir /MS3D/output/video_exports/detection_side_2min
"

# Make video from pickle:
docker exec -it ms3d_training bash -lc "
cd /MS3D/tools
export PYTHONPATH=/MS3D:/MS3D/tracker
export DISPLAY=:1

python make_video_v2.py \
  --cfg_file cfgs/dataset_configs/custom_points_dataset_da_v2.yaml \
  --ps_pkl /MS3D/tools/cfgs/target_custom/label_generation/custom_points_round1/ps_labels/final_pseudo_labels_ped_zfit_15m.pkl \
  --camera_json /MS3D/output/visualize_3d/side_cam.json \
  --splits train \
  --frame_start 0 \
  --frame_end 1200 \
  --ps_score_th 0.01 \
  --fps 10 \
  --output_dir /MS3D/output/video_exports/detection_front_2min
"
'''
import _init_path
import argparse
import copy
import datetime
import json
import os
import pickle
import subprocess
from contextlib import contextmanager
from pathlib import Path

from easydict import EasyDict
import open3d as o3d
import torch

from pcdet.config import cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from visual_utils import open3d_vis_utils as V


DEFAULT_CFG_FILE = "tools/cfgs/target_custom/ms3d_custom_round1_pv_rcnn_plusplus_points_nusc.yaml"
DEFAULT_CKPT = (
    "output/target_custom/ms3d_custom_round1_pv_rcnn_plusplus_points_nusc/"
    "nusc_points_v2_pedps_bs4_e30_3d15m/ckpt/checkpoint_epoch_30.pth"
)
DEFAULT_OUTPUT_DIR = "output/video_exports/target_make_video_v2"
REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_rgb_triplet(value):
    try:
        parts = [float(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid RGB triplet '{value}'. Expected comma-separated floats in [0, 1]."
        ) from exc

    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Invalid RGB triplet '{value}'. Expected exactly 3 comma-separated values."
        )
    if any(part < 0.0 or part > 1.0 for part in parts):
        raise argparse.ArgumentTypeError(
            f"Invalid RGB triplet '{value}'. Each component must be between 0 and 1."
        )
    return tuple(parts)


VIEW_PRESETS = {
    "front_follow": {
        "front": [-0.009079385782427972, -0.79382993606647601, 0.60807203303433444],
        "lookat": [0.13592805125144847, 25.565951040207825, -13.855443454771956],
        "up": [-0.008889802784222859, 0.60813714531420293, 0.79378220180068892],
        "zoom": 0.21999999999999992,
    },
    "front_wide": {
        "front": [-0.79490788243448329, 0.01495959113947927, 0.60654568589387037],
        "lookat": [15.417785290867977, -1.6179187751048014, -2.517384585115314],
        "up": [0.60625059182523544, -0.020154889264250107, 0.79501823900480273],
        "zoom": 0.17999999999999994,
    },
    "left_oblique": {
        "front": [0.72737973442893356, -0.51797808311597837, 0.45013045592760198],
        "lookat": [-13.773417658854088, 0.062465858514556709, -0.53706070047660459],
        "up": [-0.37595030931731882, 0.2479623453125949, 0.89284715390221758],
        "zoom": 0.079999999999999946,
    },
    "depth_camera": {
        "camera_json": "tools/DepthCamera_2026-03-25-18-20-52.json",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Batch Open3D video rendering for RoadSafeAI.")
    parser.add_argument("--cfg_file", type=str, required=True, help="Dataset or model cfg yaml.")
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint to render in ckpt mode.")
    parser.add_argument("--ps_pkl", type=str, default=None, help="Pseudo-label pickle to render in ps_pkl mode.")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Root directory for frames/videos.")
    parser.add_argument(
        "--views",
        type=str,
        default="front_follow,left_oblique,front_wide",
        help=f"Comma-separated view preset names. Available: {', '.join(sorted(VIEW_PRESETS.keys()))}",
    )
    parser.add_argument(
        "--camera_json",
        type=str,
        default=None,
        help="Optional Open3D camera parameter json. If set, it overrides --views with a single custom_camera view.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated dataset splits to render in order.",
    )
    parser.add_argument("--fps", type=int, default=10, help="Output video FPS.")
    parser.add_argument("--ffmpeg_bin", type=str, default="ffmpeg", help="ffmpeg executable path.")
    parser.add_argument("--pred_score_th", type=float, default=0.20, help="Score threshold for displayed predictions in ckpt mode.")
    parser.add_argument("--ps_score_th", type=float, default=0.40, help="Score threshold for displayed pseudo labels in ps_pkl mode.")
    parser.add_argument(
        "--class_name",
        type=str,
        default="pedestrian",
        help="Class name to keep for rendering. Use 'all' to keep every predicted class.",
    )
    parser.add_argument("--window_width", type=int, default=4000, help="Open3D render width.")
    parser.add_argument("--window_height", type=int, default=3000, help="Open3D render height.")
    parser.add_argument("--window_x", type=int, default=2000, help="Open3D window left position.")
    parser.add_argument("--window_y", type=int, default=300, help="Open3D window top position.")
    parser.add_argument("--theme", type=str, default="dark", choices=["dark", "light"], help="Render theme.")
    parser.add_argument("--point_size", type=float, default=2.0, help="Open3D point size.")
    parser.add_argument("--line_width", type=float, default=4.0, help="Open3D line width for non-linemesh boxes.")
    parser.add_argument(
        "--pred_box_color",
        type=parse_rgb_triplet,
        default=None,
        help="Optional prediction bbox color as 'r,g,b' with values in [0,1]. Example: 1,0,0 for red.",
    )
    parser.add_argument(
        "--point_color_mode",
        type=str,
        default="height",
        choices=["height", "mono"],
        help="Point cloud coloring mode.",
    )
    parser.add_argument("--use_linemesh", action="store_true", default=False, help="Use thicker line mesh boxes.")
    parser.add_argument("--use_class_colors", action="store_true", default=False, help="Color boxes by class id.")
    parser.add_argument("--frame_start", type=int, default=0, help="Global frame start index over the selected source.")
    parser.add_argument(
        "--frame_end",
        type=int,
        default=None,
        help="Global frame end index (exclusive) over the selected source. Default: render through the end.",
    )
    parser.add_argument("--workers", type=int, default=1, help="Workers for dataset build.")
    parser.add_argument("--batch_size", type=int, default=1, help="Inference batch size. Only batch_size=1 is supported.")
    return parser.parse_args()


def resolve_repo_path(path_str):
    path = Path(path_str)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    else:
        path = path.resolve()
    return path


def resolve_input_path(path_str):
    path = Path(path_str)
    if path.is_absolute():
        return path.resolve()

    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    return (REPO_ROOT / path).resolve()


def resolve_source_mode(args):
    has_ckpt = args.ckpt is not None
    has_ps_pkl = args.ps_pkl is not None
    if has_ckpt == has_ps_pkl:
        raise ValueError("Exactly one source must be provided: use either --ckpt or --ps_pkl.")
    return "ckpt" if has_ckpt else "ps_pkl"


def ensure_paths(args, source_mode):
    cfg_path = resolve_input_path(args.cfg_file)
    if not cfg_path.exists():
        raise FileNotFoundError(f"cfg_file not found: {cfg_path}")

    ckpt_path = None
    ps_pkl_path = None
    if source_mode == "ckpt":
        ckpt_path = resolve_input_path(args.ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    else:
        ps_pkl_path = resolve_input_path(args.ps_pkl)
        if not ps_pkl_path.exists():
            raise FileNotFoundError(f"ps_pkl not found: {ps_pkl_path}")

    return cfg_path, ckpt_path, ps_pkl_path


def normalize_views(args):
    if args.camera_json is not None:
        camera_json = resolve_input_path(args.camera_json)
        if not camera_json.exists():
            raise FileNotFoundError(f"camera_json not found: {camera_json}")
        return [{"name": "custom_camera", "camera_json": str(camera_json)}]

    views = []
    for view_name in [item.strip() for item in args.views.split(",") if item.strip()]:
        if view_name not in VIEW_PRESETS:
            raise ValueError(
                f"Unknown view preset: {view_name}. "
                f"Available presets: {', '.join(sorted(VIEW_PRESETS.keys()))}"
            )
        preset = copy.deepcopy(VIEW_PRESETS[view_name])
        preset["name"] = view_name
        if "camera_json" in preset:
            preset["camera_json"] = str(resolve_repo_path(preset["camera_json"]))
        views.append(preset)
    if not views:
        raise ValueError("No valid views requested.")
    return views


def resolve_window_geometry(args, views):
    window = {
        "width": args.window_width,
        "height": args.window_height,
        "x": args.window_x,
        "y": args.window_y,
        "source": "args",
    }

    camera_json_views = [view for view in views if "camera_json" in view]
    if not camera_json_views:
        return window

    camera_params = o3d.io.read_pinhole_camera_parameters(camera_json_views[0]["camera_json"])
    intrinsic = camera_params.intrinsic
    if intrinsic.width > 0 and intrinsic.height > 0:
        window["width"] = int(intrinsic.width)
        window["height"] = int(intrinsic.height)
        window["source"] = "camera_json"

    return window


def normalize_splits(args):
    valid = {"train", "val", "test"}
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    if not splits:
        raise ValueError("No dataset splits requested.")
    invalid = [split for split in splits if split not in valid]
    if invalid:
        raise ValueError(f"Unsupported split(s): {invalid}. Valid splits: {sorted(valid)}")
    return splits


def build_logger(output_dir):
    log_file = output_dir / f"make_video_v2_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    return common_utils.create_logger(log_file=log_file, rank=0)


@contextmanager
def pushd(path):
    original_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original_cwd)


def load_base_cfg(cfg_file):
    base_cfg = EasyDict()
    with pushd(REPO_ROOT / "tools"):
        cfg_from_yaml_file(str(cfg_file), base_cfg)
    return base_cfg


def resolve_data_config(base_cfg, cfg_file):
    if "dataset_configs" in str(cfg_file):
        return copy.deepcopy(base_cfg)
    if base_cfg.get("DATA_CONFIG_TAR", None):
        return copy.deepcopy(base_cfg.DATA_CONFIG_TAR)
    return copy.deepcopy(base_cfg.DATA_CONFIG)


def prepare_dataset_cfg(base_cfg, cfg_file, split):
    dataset_cfg = resolve_data_config(base_cfg, cfg_file)
    dataset_cfg.DATA_SPLIT.test = split

    if dataset_cfg.get("USE_TTA", None) is not None:
        dataset_cfg.USE_TTA = False

    if dataset_cfg.DATASET == "CustomDataset":
        if split == "train":
            dataset_cfg.INFO_PATH.train = ["custom_infos_train.pkl"]
        elif split == "val":
            dataset_cfg.INFO_PATH.val = ["custom_infos_val.pkl"]
        elif split == "test":
            dataset_cfg.INFO_PATH.test = ["custom_infos_test.pkl"]
    return dataset_cfg


def build_dataset(base_cfg, cfg_file, split, logger, workers, batch_size):
    dataset_cfg = prepare_dataset_cfg(base_cfg, cfg_file, split)
    dataset, _, _ = build_dataloader(
        dataset_cfg=dataset_cfg,
        class_names=dataset_cfg.CLASS_NAMES,
        batch_size=batch_size,
        dist=False,
        workers=workers,
        logger=logger,
        training=False,
    )
    return dataset, dataset_cfg


def collect_render_jobs(base_cfg, cfg_file, splits, logger, workers, batch_size, allowed_frame_ids=None):
    jobs = []
    split_counts = {}
    allowed_frame_ids = set(allowed_frame_ids) if allowed_frame_ids is not None else None

    for split in splits:
        dataset, dataset_cfg = build_dataset(base_cfg, cfg_file, split, logger, workers, batch_size)
        split_counts[split] = len(dataset)
        for dataset_idx, info in enumerate(dataset.infos):
            frame_id = str(info["frame_id"])
            if allowed_frame_ids is not None and frame_id not in allowed_frame_ids:
                continue
            jobs.append(
                {
                    "split": split,
                    "dataset": dataset,
                    "dataset_cfg": dataset_cfg,
                    "dataset_idx": dataset_idx,
                    "frame_id": frame_id,
                    "lidar_path": str(info["lidar_path"]),
                }
            )
    return jobs, split_counts


def slice_jobs(jobs, frame_start, frame_end):
    if frame_start < 0:
        raise ValueError("frame_start must be >= 0")
    if frame_end is not None and frame_end < frame_start:
        raise ValueError("frame_end must be >= frame_start")
    if not jobs:
        raise RuntimeError("No frames available for rendering.")
    end = len(jobs) if frame_end is None else min(frame_end, len(jobs))
    if frame_start >= len(jobs):
        raise ValueError(f"frame_start={frame_start} is outside available frame count={len(jobs)}")
    return jobs[frame_start:end]


def resolve_class_names(base_cfg, dataset_cfg):
    if base_cfg.get("CLASS_NAMES", None):
        return list(base_cfg.CLASS_NAMES)
    return list(dataset_cfg.CLASS_NAMES)


def resolve_allowed_label_ids(class_names, class_name):
    if class_name.lower() == "all":
        return None

    allowed = []
    target = class_name.lower()
    for idx, name in enumerate(class_names, start=1):
        name_lower = str(name).lower()
        if name_lower == target or target in name_lower:
            allowed.append(idx)
    if not allowed:
        raise ValueError(f"Class '{class_name}' not found in model classes: {list(class_names)}")
    return sorted(set(allowed))


def create_model(base_cfg, dataset, logger, class_names, ckpt_path):
    if not base_cfg.get("MODEL", None):
        raise ValueError("ckpt mode requires a model config with MODEL defined in --cfg_file.")
    model = build_network(model_cfg=base_cfg.MODEL, num_class=len(class_names), dataset=dataset)
    model.load_params_from_file(filename=str(ckpt_path), logger=logger, to_cpu=False)
    model.cuda()
    model.eval()
    return model


def create_visualizer(window, args):
    vis = o3d.visualization.Visualizer()
    window_ok = vis.create_window(
        window_name="RoadSafeAI make_video_v2",
        left=window["x"],
        top=window["y"],
        width=window["width"],
        height=window["height"],
        visible=True,
    )
    if not window_ok:
        display = os.environ.get("DISPLAY", "")
        raise RuntimeError(
            "Open3D window creation failed. "
            f'DISPLAY="{display}". '
            "Run with X11 forwarding and a valid display."
        )
    V.apply_render_theme(vis, theme=args.theme, point_size=args.point_size, line_width=args.line_width)
    return vis


def apply_camera(vis, view_spec):
    ctr = vis.get_view_control()
    if ctr is None:
        raise RuntimeError("Open3D view control is unavailable.")

    if "camera_json" in view_spec:
        camera_params = o3d.io.read_pinhole_camera_parameters(view_spec["camera_json"])
        try:
            ctr.convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True)
        except TypeError:
            ctr.convert_from_pinhole_camera_parameters(camera_params)
        return

    ctr.set_front(view_spec["front"])
    ctr.set_lookat(view_spec["lookat"])
    ctr.set_up(view_spec["up"])
    ctr.set_zoom(view_spec["zoom"])


def capture_views_for_frame(vis, geometries, views, frame_index, frames_root, args):
    vis.clear_geometries()
    for geom in geometries:
        vis.add_geometry(geom)
    V.apply_render_theme(vis, theme=args.theme, point_size=args.point_size, line_width=args.line_width)

    for view in views:
        apply_camera(vis, view)
        vis.poll_events()
        vis.update_renderer()
        output_path = frames_root / view["name"] / f"{frame_index:06d}.png"
        vis.capture_screen_image(str(output_path), do_render=True)


def make_geometries(points, pred_dict, args):
    geom_kwargs = {
        "use_linemesh": args.use_linemesh,
        "use_class_colors": args.use_class_colors,
        "theme": args.theme,
        "point_color_mode": args.point_color_mode,
        "ref_box_colors": args.pred_box_color,
    }
    return V.get_geometries(
        points=points,
        ref_boxes=pred_dict["pred_boxes"],
        ref_scores=pred_dict["pred_scores"],
        ref_labels=pred_dict["pred_labels"],
        **geom_kwargs,
    )


def build_label_mask(labels, allowed_label_ids):
    if allowed_label_ids is None:
        return torch.ones(labels.shape, dtype=torch.bool, device=labels.device)

    mask = torch.zeros_like(labels, dtype=torch.bool)
    for label_id in allowed_label_ids:
        mask |= labels == int(label_id)
    return mask


def prepare_frames_root(output_dir, views):
    frames_root = output_dir / "frames"
    for view in views:
        (frames_root / view["name"]).mkdir(parents=True, exist_ok=True)
    return frames_root


def render_frames_from_ckpt(model, jobs, views, args, output_dir, logger, allowed_label_ids):
    frames_root = prepare_frames_root(output_dir, views)
    rendered = []
    vis = create_visualizer(args.window, args)
    try:
        with torch.no_grad():
            for global_idx, job in enumerate(jobs):
                dataset = job["dataset"]
                sample = dataset[job["dataset_idx"]]
                batch_dict = dataset.collate_batch([sample])
                load_data_to_gpu(batch_dict)
                pred_dicts, _ = model.forward(batch_dict)
                pred_dict = pred_dicts[0]

                pred_scores = pred_dict["pred_scores"]
                pred_labels = pred_dict["pred_labels"]
                score_mask = pred_scores >= args.pred_score_th
                score_mask &= build_label_mask(pred_labels, allowed_label_ids)

                pred_dict_filtered = {
                    "pred_boxes": pred_dict["pred_boxes"][score_mask],
                    "pred_scores": pred_dict["pred_scores"][score_mask],
                    "pred_labels": pred_dict["pred_labels"][score_mask],
                }
                geometries = make_geometries(batch_dict["points"][:, 1:], pred_dict_filtered, args)
                capture_views_for_frame(vis, geometries, views, global_idx, frames_root, args)

                shown_boxes = int(pred_dict_filtered["pred_boxes"].shape[0])
                logger.info(
                    "[%04d/%04d][ckpt] split=%s dataset_idx=%d frame_id=%s shown_boxes=%d",
                    global_idx + 1,
                    len(jobs),
                    job["split"],
                    job["dataset_idx"],
                    job["frame_id"],
                    shown_boxes,
                )
                rendered.append(
                    {
                        "global_index": global_idx,
                        "split": job["split"],
                        "dataset_idx": job["dataset_idx"],
                        "frame_id": job["frame_id"],
                        "lidar_path": job["lidar_path"],
                        "shown_boxes": shown_boxes,
                    }
                )
    finally:
        vis.destroy_window()
    return rendered


def load_ps_dict(ps_pkl_path):
    with open(ps_pkl_path, "rb") as f:
        ps_dict = pickle.load(f)
    if not isinstance(ps_dict, dict):
        raise ValueError(f"Expected ps_pkl to load a dict keyed by frame_id, got {type(ps_dict)}")
    return {str(frame_id): value for frame_id, value in ps_dict.items()}


def get_ps_predictions(ps_dict, frame_id):
    if frame_id not in ps_dict:
        raise KeyError(f"frame_id={frame_id} not found in ps_pkl")
    entry = ps_dict[frame_id]
    if "gt_boxes" not in entry:
        raise KeyError(f"frame_id={frame_id} in ps_pkl is missing 'gt_boxes'")

    gt_boxes = torch.as_tensor(entry["gt_boxes"])
    if gt_boxes.ndim != 2 or gt_boxes.shape[1] < 9:
        raise ValueError(
            f"frame_id={frame_id} gt_boxes must have shape [N, >=9], got {tuple(gt_boxes.shape)}"
        )

    return {
        "pred_boxes": gt_boxes[:, :7].float(),
        "pred_labels": gt_boxes[:, 7].abs().to(torch.int64),
        "pred_scores": gt_boxes[:, 8].float(),
    }


def render_frames_from_ps_pkl(ps_dict, jobs, views, args, output_dir, logger, allowed_label_ids):
    frames_root = prepare_frames_root(output_dir, views)
    rendered = []
    vis = create_visualizer(args.window, args)
    try:
        for global_idx, job in enumerate(jobs):
            dataset = job["dataset"]
            sample = dataset[job["dataset_idx"]]
            pred_dict = get_ps_predictions(ps_dict, job["frame_id"])
            score_mask = pred_dict["pred_scores"] >= args.ps_score_th
            score_mask &= build_label_mask(pred_dict["pred_labels"], allowed_label_ids)

            pred_dict_filtered = {
                "pred_boxes": pred_dict["pred_boxes"][score_mask],
                "pred_scores": pred_dict["pred_scores"][score_mask],
                "pred_labels": pred_dict["pred_labels"][score_mask],
            }
            geometries = make_geometries(torch.as_tensor(sample["points"]).float(), pred_dict_filtered, args)
            capture_views_for_frame(vis, geometries, views, global_idx, frames_root, args)

            shown_boxes = int(pred_dict_filtered["pred_boxes"].shape[0])
            logger.info(
                "[%04d/%04d][ps_pkl] split=%s dataset_idx=%d frame_id=%s shown_boxes=%d",
                global_idx + 1,
                len(jobs),
                job["split"],
                job["dataset_idx"],
                job["frame_id"],
                shown_boxes,
            )
            rendered.append(
                {
                    "global_index": global_idx,
                    "split": job["split"],
                    "dataset_idx": job["dataset_idx"],
                    "frame_id": job["frame_id"],
                    "lidar_path": job["lidar_path"],
                    "shown_boxes": shown_boxes,
                }
            )
    finally:
        vis.destroy_window()
    return rendered


def run_ffmpeg(cmd, cwd=None):
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg command failed:\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


def raise_if_frames_missing(views, output_dir):
    for view in views:
        frame_dir = output_dir / "frames" / view["name"]
        if not any(frame_dir.glob("*.png")):
            raise RuntimeError(f"No rendered PNG frames found for view '{view['name']}' in {frame_dir}")


def build_per_view_videos(views, output_dir, fps, ffmpeg_bin):
    raise_if_frames_missing(views, output_dir)
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    per_view_videos = []
    for view in views:
        frames_pattern = output_dir / "frames" / view["name"] / "%06d.png"
        view_video_path = videos_dir / f"{view['name']}.mp4"
        ffmpeg_cmd = [
            ffmpeg_bin,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frames_pattern),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(view_video_path),
        ]
        run_ffmpeg(ffmpeg_cmd)
        per_view_videos.append(view_video_path)
    return per_view_videos


def concat_videos(per_view_videos, output_dir, ffmpeg_bin):
    concat_list = output_dir / "videos" / "concat_list.txt"
    concat_list.write_text("".join([f"file '{video.resolve()}'\n" for video in per_view_videos]))
    final_video = output_dir / "videos" / "final_concat.mp4"
    ffmpeg_cmd = [
        ffmpeg_bin,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list),
        "-c",
        "copy",
        str(final_video),
    ]
    run_ffmpeg(ffmpeg_cmd)
    return final_video, concat_list


def write_metadata(
    output_dir,
    args,
    source_mode,
    cfg_path,
    ckpt_path,
    ps_pkl_path,
    views,
    splits,
    split_counts,
    selected_jobs,
    rendered_frames,
    per_view_videos,
    final_video,
):
    metadata = {
        "created_at": datetime.datetime.now().isoformat(),
        "source_mode": source_mode,
        "cfg_file": str(cfg_path),
        "ckpt": str(ckpt_path) if ckpt_path is not None else None,
        "ps_pkl": str(ps_pkl_path) if ps_pkl_path is not None else None,
        "splits": splits,
        "split_counts": split_counts,
        "selected_global_frame_start": args.frame_start,
        "selected_global_frame_end": len(selected_jobs) + args.frame_start,
        "selected_frame_count": len(selected_jobs),
        "views": views,
        "fps": args.fps,
        "pred_score_th": args.pred_score_th if source_mode == "ckpt" else None,
        "ps_score_th": args.ps_score_th if source_mode == "ps_pkl" else None,
        "class_name": args.class_name,
        "theme": args.theme,
        "point_size": args.point_size,
        "line_width": args.line_width,
        "pred_box_color": list(args.pred_box_color) if args.pred_box_color is not None else None,
        "window": args.window,
        "per_view_videos": [str(path) for path in per_view_videos],
        "final_video": str(final_video),
        "rendered_frames": rendered_frames,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return metadata_path


def ensure_ffmpeg(ffmpeg_bin):
    try:
        subprocess.run([ffmpeg_bin, "-version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"ffmpeg is required but '{ffmpeg_bin}' was not found or failed to start. "
            "Install ffmpeg in the runtime environment or pass --ffmpeg_bin."
        ) from exc


def main():
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("Only batch_size=1 is supported for deterministic Open3D rendering.")

    source_mode = resolve_source_mode(args)
    cfg_path, ckpt_path, ps_pkl_path = ensure_paths(args, source_mode)
    views = normalize_views(args)
    args.window = resolve_window_geometry(args, views)
    splits = normalize_splits(args)
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = build_logger(output_dir)
    logger.info("Starting make_video_v2")
    logger.info("source_mode=%s", source_mode)
    logger.info("cfg_file=%s", cfg_path)
    logger.info("ckpt=%s", ckpt_path)
    logger.info("ps_pkl=%s", ps_pkl_path)
    logger.info("window=%s", args.window)
    logger.info("output_dir=%s", output_dir)

    ensure_ffmpeg(args.ffmpeg_bin)

    base_cfg = load_base_cfg(cfg_path)
    ps_dict = None
    allowed_frame_ids = None
    if source_mode == "ps_pkl":
        ps_dict = load_ps_dict(ps_pkl_path)
        allowed_frame_ids = set(ps_dict.keys())

    jobs, split_counts = collect_render_jobs(
        base_cfg=base_cfg,
        cfg_file=cfg_path,
        splits=splits,
        logger=logger,
        workers=args.workers,
        batch_size=args.batch_size,
        allowed_frame_ids=allowed_frame_ids,
    )
    selected_jobs = slice_jobs(jobs, args.frame_start, args.frame_end)
    dataset_cfg = selected_jobs[0]["dataset_cfg"]
    class_names = resolve_class_names(base_cfg, dataset_cfg)
    allowed_label_ids = resolve_allowed_label_ids(class_names, args.class_name)
    logger.info("Allowed label ids: %s", "all" if allowed_label_ids is None else allowed_label_ids)

    if source_mode == "ckpt":
        dataset_for_model = selected_jobs[0]["dataset"]
        model = create_model(base_cfg, dataset_for_model, logger, class_names, ckpt_path)
        rendered_frames = render_frames_from_ckpt(model, selected_jobs, views, args, output_dir, logger, allowed_label_ids)
    else:
        rendered_frames = render_frames_from_ps_pkl(ps_dict, selected_jobs, views, args, output_dir, logger, allowed_label_ids)

    per_view_videos = build_per_view_videos(views, output_dir, args.fps, args.ffmpeg_bin)
    final_video, _ = concat_videos(per_view_videos, output_dir, args.ffmpeg_bin)
    metadata_path = write_metadata(
        output_dir=output_dir,
        args=args,
        source_mode=source_mode,
        cfg_path=cfg_path,
        ckpt_path=ckpt_path,
        ps_pkl_path=ps_pkl_path,
        views=views,
        splits=splits,
        split_counts=split_counts,
        selected_jobs=selected_jobs,
        rendered_frames=rendered_frames,
        per_view_videos=per_view_videos,
        final_video=final_video,
    )

    logger.info("Done. Final video: %s", final_video)
    logger.info("Metadata: %s", metadata_path)
    print(f"Final video written to: {final_video}")
    print(f"Metadata written to: {metadata_path}")


if __name__ == "__main__":
    main()
