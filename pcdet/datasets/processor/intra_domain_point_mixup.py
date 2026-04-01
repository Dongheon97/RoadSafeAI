import copy
import numpy as np

def shuffle_points(data_dict):
    points = data_dict['points']
    shuffle_idx = np.random.permutation(points.shape[0])
    data_dict['points'] = points[shuffle_idx]
    return data_dict

def intra_domain_point_mixup(data_dict_1, data_dict_2, alpha=2.0):
    new_data_dict = copy.deepcopy(data_dict_1)
    lam = np.random.beta(alpha, alpha)
    data_dict_1 = shuffle_points(data_dict_1)
    data_dict_2 = shuffle_points(data_dict_2)
    pts_1_len = int(data_dict_1['points'].shape[0] * lam)
    pts_2_len = int(data_dict_2['points'].shape[0] * (1 - lam))
    new_data_dict['points'] = np.concatenate((data_dict_1['points'][:pts_1_len], data_dict_2['points'][:pts_2_len]), axis=0)
    if 'gt_boxes' in data_dict_1 and 'gt_boxes' in data_dict_2:
        if len(data_dict_1['gt_boxes']) > 0 and len(data_dict_2['gt_boxes']) > 0:
            new_data_dict['gt_boxes'] = np.concatenate((data_dict_1['gt_boxes'], data_dict_2['gt_boxes']), axis=0)
        elif len(data_dict_2['gt_boxes']) > 0:
            new_data_dict['gt_boxes'] = data_dict_2['gt_boxes']
    return new_data_dict
