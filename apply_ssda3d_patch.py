import os

os.makedirs('pcdet/datasets/processor', exist_ok=True)
with open('pcdet/datasets/processor/intra_domain_point_mixup.py', 'w') as f:
    f.write("""import copy\nimport numpy as np\n\ndef shuffle_points(data_dict):\n    points = data_dict['points']\n    shuffle_idx = np.random.permutation(points.shape[0])\n    data_dict['points'] = points[shuffle_idx]\n    return data_dict\n\ndef intra_domain_point_mixup(data_dict_1, data_dict_2, alpha=2.0):\n    new_data_dict = copy.deepcopy(data_dict_1)\n    lam = np.random.beta(alpha, alpha)\n    data_dict_1 = shuffle_points(data_dict_1)\n    data_dict_2 = shuffle_points(data_dict_2)\n    pts_1_len = int(data_dict_1['points'].shape[0] * lam)\n    pts_2_len = int(data_dict_2['points'].shape[0] * (1 - lam))\n    new_data_dict['points'] = np.concatenate((data_dict_1['points'][:pts_1_len], data_dict_2['points'][:pts_2_len]), axis=0)\n    if 'gt_boxes' in data_dict_1 and 'gt_boxes' in data_dict_2:\n        if len(data_dict_1['gt_boxes']) > 0 and len(data_dict_2['gt_boxes']) > 0:\n            new_data_dict['gt_boxes'] = np.concatenate((data_dict_1['gt_boxes'], data_dict_2['gt_boxes']), axis=0)\n        elif len(data_dict_2['gt_boxes']) > 0:\n            new_data_dict['gt_boxes'] = data_dict_2['gt_boxes']\n    return new_data_dict\n""")

os.makedirs('pcdet/datasets/custom', exist_ok=True)
with open('pcdet/datasets/custom/custom_mixup_dataset.py', 'w') as f:
    f.write("""import copy\nimport pickle\nimport numpy as np\nfrom pathlib import Path\nfrom .custom_dataset import CustomDataset\nfrom ..processor.intra_domain_point_mixup import intra_domain_point_mixup\n\nclass CustomMixUpDataset(CustomDataset):\n    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, pseudo_info_path=None):\n        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger)\n        self.pseudo_info_path = pseudo_info_path\n        self.gt_infos = copy.deepcopy(self.infos)\n        self.ps_infos = []\n        if self.pseudo_info_path and Path(self.pseudo_info_path).exists():\n            with open(self.pseudo_info_path, 'rb') as f:\n                self.ps_infos = pickle.load(f)\n        self.infos = self.gt_infos + self.ps_infos\n        if logger:\n            logger.info(f"CustomMixUpDataset 載入: {len(self.gt_infos)} GT, {len(self.ps_infos)} Pseudo.")\n\n    def get_data_dict_from_info(self, info):\n        pc_info = info['point_cloud']\n        sequence_name = pc_info['lidar_idx']\n        points = self.get_lidar(sequence_name)\n        input_dict = {'points': points, 'frame_id': sequence_name}\n        if 'annos' in info:\n            annos = info['annos']\n            input_dict.update({'gt_names': annos['name'], 'gt_boxes': annos['gt_boxes_lidar']})\n        return input_dict\n\n    def prepare_mixup_data(self, dict_1, dict_2):\n        if self.training:\n            mask_1 = np.ones(len(dict_1.get('gt_names', [])), dtype=bool)\n            dict_1 = self.data_augmentor.forward(data_dict={**dict_1, 'gt_boxes_mask': mask_1})\n            mask_2 = np.ones(len(dict_2.get('gt_names', [])), dtype=bool)\n            dict_2 = self.data_augmentor.forward(data_dict={**dict_2, 'gt_boxes_mask': mask_2})\n        for d in [dict_1, dict_2]:\n            if len(d.get('gt_boxes', [])) > 0:\n                gt_classes = np.array([self.class_names.index(n) + 1 for n in d['gt_names']], dtype=np.int32)\n                d['gt_boxes'] = np.concatenate((d['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)\n            if d.get('points', None) is not None:\n                d = self.point_feature_encoder.forward(d)\n        mixed_dict = intra_domain_point_mixup(dict_1, dict_2, alpha=self.dataset_cfg.ALPHA)\n        mixed_dict = self.data_processor.forward(data_dict=mixed_dict)\n        mixed_dict.pop('gt_names', None)\n        return mixed_dict\n\n    def __getitem__(self, index):\n        prob = np.random.random(1)\n        if prob > self.dataset_cfg.MIXUP_PROB:\n            is_gt = np.random.random(1) < self.dataset_cfg.GT_PROB\n            if is_gt or len(self.ps_infos) == 0:\n                info = copy.deepcopy(self.gt_infos[np.random.randint(len(self.gt_infos))])\n            else:\n                info = copy.deepcopy(self.ps_infos[np.random.randint(len(self.ps_infos))])\n            return self.prepare_data(self.get_data_dict_from_info(info))\n        else:\n            idx1 = np.random.randint(len(self.gt_infos))\n            idx2 = np.random.randint(len(self.ps_infos)) if len(self.ps_infos) > 0 else np.random.randint(len(self.gt_infos))\n            dict_1 = self.get_data_dict_from_info(copy.deepcopy(self.gt_infos[idx1]))\n            dict_2 = self.get_data_dict_from_info(copy.deepcopy(self.ps_infos[idx2]) if len(self.ps_infos) > 0 else copy.deepcopy(self.gt_infos[idx2]))\n            return self.prepare_mixup_data(dict_1, dict_2)\n""")

init_file = 'pcdet/datasets/__init__.py'
with open(init_file, 'r') as f:
    content = f.read()

if 'build_mixup_dataloader' not in content:
    patch = """
from .custom.custom_mixup_dataset import CustomMixUpDataset
__all__['CustomMixUpDataset'] = CustomMixUpDataset

def build_mixup_dataloader(dataset_cfg, class_names, batch_size, dist, root_path=None, workers=4, logger=None, training=True, pseudo_info_path=None):
    import torch
    from torch.utils.data import DataLoader
    dataset = __all__[dataset_cfg.DATASET_NAME](
        dataset_cfg=dataset_cfg, class_names=class_names, root_path=root_path,
        training=training, logger=logger, pseudo_info_path=pseudo_info_path
    )
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if dist else None
    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, num_workers=workers,
        shuffle=(sampler is None) and training, collate_fn=dataset.collate_batch, drop_last=False, sampler=sampler, timeout=0
    )
    return dataset, dataloader, sampler
"""
    with open(init_file, 'a') as f:
        f.write(patch)
    print("✅ 成功注入 SSDA3D 的 CustomMixUpDataset 與 Dataloader！")
else:
    print("⚡ 模組似乎已經注入過了。")
