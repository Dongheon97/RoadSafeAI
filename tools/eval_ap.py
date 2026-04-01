import pickle
import numpy as np
import torch
import sys
import os
sys.path.append('..')
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu

def compute_global_ap(gt_infos, det_annos, class_name, iou_thresh):
    all_det_boxes, all_det_scores, all_det_frame_ids = [], [], []
    frame_gt_boxes, frame_gt_matched = {}, {}
    total_gt = 0
    
    for i in range(len(gt_infos)):
        gt_anno = gt_infos[i]['annos']
        gt_mask = (gt_anno['name'] == class_name)
        gt_boxes = gt_anno['gt_boxes_lidar'][gt_mask] if 'gt_boxes_lidar' in gt_anno else np.zeros((0,7))
        frame_gt_boxes[i] = gt_boxes
        frame_gt_matched[i] = np.zeros(len(gt_boxes), dtype=bool)
        total_gt += len(gt_boxes)
        
        det = det_annos[i]
        det_mask = (det['name'] == class_name)
        det_boxes = det['boxes_lidar'][det_mask]
        det_scores = det['score'][det_mask]
        
        for b, s in zip(det_boxes, det_scores):
            all_det_boxes.append(b)
            all_det_scores.append(s)
            all_det_frame_ids.append(i)
            
    if total_gt == 0 or len(all_det_boxes) == 0: 
        return 0.0
    
    all_det_boxes = np.array(all_det_boxes)
    all_det_scores = np.array(all_det_scores)
    all_det_frame_ids = np.array(all_det_frame_ids)
    
    # Sort detections by confidence score
    indices = np.argsort(-all_det_scores)
    all_det_boxes, all_det_scores, all_det_frame_ids = all_det_boxes[indices], all_det_scores[indices], all_det_frame_ids[indices]
    
    nd = len(all_det_boxes)
    tp, fp = np.zeros(nd), np.zeros(nd)
    
    for d in range(nd):
        frame_id = all_det_frame_ids[d]
        det_box = all_det_boxes[d:d+1]
        gt_boxes = frame_gt_boxes[frame_id]
        
        if len(gt_boxes) == 0:
            fp[d] = 1
            continue
            
        det_tensor = torch.from_numpy(det_box).cuda().float()
        gt_tensor = torch.from_numpy(gt_boxes).cuda().float()
        ious = boxes_iou3d_gpu(det_tensor, gt_tensor).cpu().numpy()[0]
        
        best_iou, best_g = 0, -1
        for g in range(len(gt_boxes)):
            if ious[g] > best_iou:
                best_iou = ious[g]
                best_g = g
                
        if best_iou >= iou_thresh:
            if not frame_gt_matched[frame_id][best_g]:
                tp[d] = 1
                frame_gt_matched[frame_id][best_g] = True
            else:
                fp[d] = 1
        else:
            fp[d] = 1
            
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    rec = tp / total_gt
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(rec >= t) == 0:
            p = 0
        else:
            p = np.max(prec[rec >= t])
        ap += p / 11.0
    return ap

if __name__ == '__main__':
    with open('../data/custom/custom_infos_val.pkl', 'rb') as f:
        gt_infos = pickle.load(f)

    res_file = '../output/custom_models/pv_rcnn/default/eval/epoch_80/val/default/result.pkl'
    if not os.path.exists(res_file):
        print(f"Error: {res_file} not found!")
        sys.exit(1)

    with open(res_file, 'rb') as f:
        det_annos = pickle.load(f)

    class_names = ['Vehicle', 'Pedestrian', 'Cyclist']
    iou_thresholds = [0.3, 0.5, 0.7]

    print("="*50)
    print("🚀 CUSTOM 3D AP EVALUATION (LiDAR Coords) 🚀")
    print("="*50)

    for cls in class_names:
        print(f"\n--- Class: {cls} ---")
        for iou_thresh in iou_thresholds:
            ap = compute_global_ap(gt_infos, det_annos, cls, iou_thresh)
            print(f"AP@{iou_thresh} : {ap*100:.2f}%")
    print("\n" + "="*50)
