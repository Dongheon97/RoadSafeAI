import copy
import pickle
from pathlib import Path

import numpy as np

from ..dataset import DatasetTemplate


class CustomDataset(DatasetTemplate):
    """
    Custom target dataset loader for flat OpenPCDet-style layout:
      data/custom_dataset/
        training/velodyne/*.bin
        ImageSets/{train,val,test}.txt
        custom_infos_{train,val}.pkl
    """

    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.infos = []
        self.frameid_to_idx = {}
        self.seq_name_to_infos = {}
        self.seq_name_to_len = {}
        self.seq_name_to_sample = {}
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.include_data()

    def include_data(self):
        if self.logger is not None:
            self.logger.info("Loading CustomDataset")

        custom_infos = []
        for info_path in self.dataset_cfg.INFO_PATH[self.split]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, "rb") as f:
                infos = pickle.load(f)
                custom_infos.extend(infos)

        # Optional downsampling for large datasets.
        sampled_interval = 1
        if self.dataset_cfg.get("SAMPLED_INTERVAL", None):
            sampled_interval = int(self.dataset_cfg.SAMPLED_INTERVAL[self.mode])
        if sampled_interval > 1:
            custom_infos = [custom_infos[i] for i in range(0, len(custom_infos), sampled_interval)]

        self.infos = custom_infos
        self.seq_name_to_infos = {}
        self.seq_name_to_sample = {}

        for idx, info in enumerate(self.infos):
            frame_id = str(info.get("frame_id", idx))
            info["frame_id"] = frame_id
            self.frameid_to_idx[frame_id] = idx

            pc_info = info.setdefault("point_cloud", {})
            seq_name = str(pc_info.get("lidar_sequence", "custom_seq_0"))
            sample_idx = int(pc_info.get("sample_idx", idx))
            pc_info["lidar_sequence"] = seq_name
            pc_info["sample_idx"] = sample_idx
            pc_info.setdefault("lidar_idx", frame_id)
            pc_info.setdefault("num_features", 4)

            info.setdefault("timestamp", sample_idx)
            if "pose" not in info:
                info["pose"] = np.eye(4, dtype=np.float32)
            info["pose"] = np.asarray(info["pose"], dtype=np.float32).reshape(4, 4)

            # Keep raw info format but normalize for compatibility utilities.
            if info.get("annos", None) is None:
                info["annos"] = {"name": np.array([]), "gt_boxes_lidar": np.empty((0, 7), dtype=np.float32)}

            if seq_name not in self.seq_name_to_infos:
                self.seq_name_to_infos[seq_name] = []
            self.seq_name_to_infos[seq_name].append(info)

        self.seq_name_to_len = {}
        for seq_name, seq_infos in self.seq_name_to_infos.items():
            seq_infos.sort(key=lambda x: x["point_cloud"]["sample_idx"])
            self.seq_name_to_len[seq_name] = len(seq_infos)
            self.seq_name_to_sample[seq_name] = {
                int(x["point_cloud"]["sample_idx"]): x for x in seq_infos
            }

        if self.logger is not None:
            self.logger.info("Total samples for CustomDataset: %d", len(self.infos))

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs
        return len(self.infos)

    @staticmethod
    def remove_ego_points(points, center_radius=1.0):
        mask = ~((np.abs(points[:, 0]) < center_radius) & (np.abs(points[:, 1]) < center_radius))
        return points[mask]

    def get_lidar(self, lidar_path):
        lidar_path = Path(lidar_path)
        if not lidar_path.is_absolute():
            lidar_path = self.root_path / lidar_path

        points = np.fromfile(str(lidar_path), dtype=np.float32)
        if points.size == 0:
            return np.zeros((0, 4), dtype=np.float32)

        for dim in (4, 5, 3):
            if points.size % dim == 0:
                pts = points.reshape(-1, dim)
                if dim >= 4:
                    return pts[:, :4].astype(np.float32)
                xyz = pts[:, :3].astype(np.float32)
                return np.hstack([xyz, np.zeros((xyz.shape[0], 1), dtype=np.float32)])
        raise ValueError(f"Cannot infer point dimension for {lidar_path}")

    def get_sequence_data(self, info, points, sequence_name, sample_idx, max_sweeps):
        points = self.remove_ego_points(points, center_radius=1.5)
        points = np.hstack([points, np.zeros((points.shape[0], 1), dtype=points.dtype)])  # + timestamp

        if max_sweeps <= 1:
            return points

        sample_map = self.seq_name_to_sample.get(sequence_name, {})
        if sample_idx not in sample_map:
            return points

        pose_cur = np.asarray(info["pose"], dtype=np.float32).reshape(4, 4)
        prev_points_all = []
        prev_indices = np.arange(max(0, sample_idx - int(max_sweeps) + 1), sample_idx, dtype=np.int32)
        for sample_idx_pre in prev_indices:
            prev_info = sample_map.get(int(sample_idx_pre))
            if prev_info is None:
                continue
            points_pre = self.get_lidar(prev_info["lidar_path"])
            points_pre = self.remove_ego_points(points_pre, center_radius=1.5)

            pose_pre = np.asarray(prev_info["pose"], dtype=np.float32).reshape(4, 4)
            expand_points_pre = np.concatenate([points_pre[:, :3], np.ones((points_pre.shape[0], 1), dtype=np.float32)], axis=1)
            points_pre_global = (expand_points_pre @ pose_pre.T)[:, :3]
            expand_points_pre_global = np.concatenate(
                [points_pre_global, np.ones((points_pre_global.shape[0], 1), dtype=np.float32)], axis=1
            )
            points_pre_cur = (expand_points_pre_global @ np.linalg.inv(pose_cur).T)[:, :3]

            cur = np.concatenate([points_pre_cur, points_pre[:, 3:4]], axis=1)
            rel_t = 0.1 * (sample_idx - sample_idx_pre) * np.ones((cur.shape[0], 1), dtype=np.float32)
            cur = np.hstack([cur, rel_t])
            prev_points_all.append(cur)

        if prev_points_all:
            points = np.concatenate([points] + prev_points_all, axis=0).astype(np.float32)
        return points

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = copy.deepcopy(self.infos[index])
        pc_info = info["point_cloud"]
        frame_id = str(info["frame_id"])
        sequence_name = pc_info["lidar_sequence"]
        sample_idx = int(pc_info["sample_idx"])

        points = self.get_lidar(info["lidar_path"])
        points = self.get_sequence_data(
            info=info,
            points=points,
            sequence_name=sequence_name,
            sample_idx=sample_idx,
            max_sweeps=int(self.dataset_cfg.get("MAX_SWEEPS", 1)),
        )

        if self.dataset_cfg.get("SHIFT_COOR", None):
            points[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)

        input_dict = {
            "points": points,
            "frame_id": frame_id,
            "sample_idx": sample_idx,
            "gt_boxes": None if self.training else np.empty((0, 7), dtype=np.float32),
            "gt_names": np.empty((0)),
        }

        if self.dataset_cfg.get("USE_PSEUDO_LABEL", None) and self.training:
            # Pseudo-label class IDs are fixed as 1:Vehicle 2:Pedestrian 3:Cyclist.
            psid2clsid = {}
            if "Vehicle" in self.class_names:
                psid2clsid[1] = self.class_names.index("Vehicle") + 1
            if "Pedestrian" in self.class_names:
                psid2clsid[2] = self.class_names.index("Pedestrian") + 1
            if "Cyclist" in self.class_names:
                psid2clsid[3] = self.class_names.index("Cyclist") + 1

            # Support lower-case class naming as well.
            if "car" in self.class_names and 1 not in psid2clsid:
                psid2clsid[1] = self.class_names.index("car") + 1
            if "pedestrian" in self.class_names and 2 not in psid2clsid:
                psid2clsid[2] = self.class_names.index("pedestrian") + 1
            if "cyclist" in self.class_names and 3 not in psid2clsid:
                psid2clsid[3] = self.class_names.index("cyclist") + 1

            self.fill_pseudo_labels(input_dict, psid2clsid)

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

    def evaluation(self, det_annos, class_names, **kwargs):
        # Custom target data is typically unlabeled.
        if len(self.infos) == 0:
            return None, {}
        annos = self.infos[0].get("annos", None)
        if annos is None or len(annos.get("gt_boxes_lidar", [])) == 0:
            if self.logger is not None:
                self.logger.info("Skipping evaluation - no gt annotations provided.")
            return None, {}
        raise NotImplementedError

