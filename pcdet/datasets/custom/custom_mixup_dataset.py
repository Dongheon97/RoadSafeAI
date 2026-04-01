import copy
import pickle
import numpy as np
from pathlib import Path
from .custom_dataset import CustomDataset
from ..processor.intra_domain_point_mixup import intra_domain_point_mixup

class CustomMixUpDataset(CustomDataset):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, pseudo_info_path=None):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger)
        self.pseudo_info_path = pseudo_info_path
        
        if hasattr(self, 'custom_infos'):
            self.gt_infos = copy.deepcopy(self.custom_infos)
        else:
            self.gt_infos = copy.deepcopy(self.infos)
            
        self.ps_infos = []
        if self.pseudo_info_path and Path(self.pseudo_info_path).exists():
            with open(self.pseudo_info_path, 'rb') as f:
                self.ps_infos = pickle.load(f)
                
        self.custom_infos = self.gt_infos + self.ps_infos
        self.infos = self.custom_infos
        
        if logger:
            logger.info(f"CustomMixUpDataset loaded: include {len(self.gt_infos)} GT, {len(self.ps_infos)}")

    def get_data_dict_from_info(self, info):
        pc_info = info['point_cloud']
        sequence_name = pc_info['lidar_idx']
        points = self.get_lidar(sequence_name)
        input_dict = {'points': points, 'frame_id': sequence_name}
        if 'annos' in info:
            annos = info['annos']
            input_dict.update({'gt_names': annos['name'], 'gt_boxes': annos['gt_boxes_lidar']})
        return input_dict

    def prepare_mixup_data(self, dict_1, dict_2):
        if self.training:
            mask_1 = np.ones(len(dict_1.get('gt_names', [])), dtype=bool)
            dict_1 = self.data_augmentor.forward(data_dict={**dict_1, 'gt_boxes_mask': mask_1})
            mask_2 = np.ones(len(dict_2.get('gt_names', [])), dtype=bool)
            dict_2 = self.data_augmentor.forward(data_dict={**dict_2, 'gt_boxes_mask': mask_2})
            
        for d in [dict_1, dict_2]:
            if len(d.get('gt_boxes', [])) > 0:
                gt_classes = np.array([self.class_names.index(n) + 1 for n in d['gt_names']], dtype=np.int32)
                d['gt_boxes'] = np.concatenate((d['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            if d.get('points', None) is not None:
                d = self.point_feature_encoder.forward(d)
                
        mixed_dict = intra_domain_point_mixup(dict_1, dict_2, alpha=self.dataset_cfg.ALPHA)

        if len(mixed_dict['gt_boxes'].shape) != 2:
            return self.__getitem__(np.random.randint(self.__len__()))
            
        mixed_dict = self.data_processor.forward(data_dict=mixed_dict)
        
        if self.training and len(mixed_dict['gt_boxes']) == 0:
            return self.__getitem__(np.random.randint(self.__len__()))
            
        mixed_dict.pop('gt_names', None)
        return mixed_dict

    def __getitem__(self, index):
        prob = np.random.random(1)
        if prob > self.dataset_cfg.MIXUP_PROB:
            is_gt = np.random.random(1) < self.dataset_cfg.GT_PROB
            if is_gt or len(self.ps_infos) == 0:
                info = copy.deepcopy(self.gt_infos[np.random.randint(len(self.gt_infos))])
            else:
                info = copy.deepcopy(self.ps_infos[np.random.randint(len(self.ps_infos))])
            return self.prepare_data(self.get_data_dict_from_info(info))
        else:
            idx1 = np.random.randint(len(self.gt_infos))
            idx2 = np.random.randint(len(self.ps_infos)) if len(self.ps_infos) > 0 else np.random.randint(len(self.gt_infos))
            dict_1 = self.get_data_dict_from_info(copy.deepcopy(self.gt_infos[idx1]))
            dict_2 = self.get_data_dict_from_info(copy.deepcopy(self.ps_infos[idx2]) if len(self.ps_infos) > 0 else copy.deepcopy(self.gt_infos[idx2]))
            return self.prepare_mixup_data(dict_1, dict_2)
