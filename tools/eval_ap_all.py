#!/usr/bin/env python3
"""
Ablation Study Evaluation & Visualization
==========================================
Computes 3D AP for Pedestrian across all experiment groups,
generates comparison bar charts and a summary table.

Usage:
  cd ~/RoadSafeAI/tools
  python eval_ablation.py

Output:
  - Terminal: formatted table with AP@0.3/0.5/0.7 + IoU + Recall
  - PNG: ablation_results.png (bar chart)
"""

import pickle
import numpy as np
import torch
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict

sys.path.append('..')
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu


# ============================================================
# Core Metrics
# ============================================================

def compute_per_frame_stats(gt_boxes, det_boxes, det_scores, iou_thresh):
    """Compute TP/FP/FN and IoU for a single frame."""
    if len(det_boxes) == 0:
        return 0, 0, len(gt_boxes), []
    if len(gt_boxes) == 0:
        return 0, len(det_boxes), 0, []

    det_tensor = torch.from_numpy(det_boxes).cuda().float()
    gt_tensor = torch.from_numpy(gt_boxes).cuda().float()
    ious = boxes_iou3d_gpu(det_tensor, gt_tensor).cpu().numpy()

    gt_matched = np.zeros(len(gt_boxes), dtype=bool)
    tp, fp = 0, 0
    matched_ious = []

    # Sort detections by score (descending)
    order = np.argsort(-det_scores)
    for d in order:
        best_iou = ious[d].max() if len(ious[d]) > 0 else 0
        best_g = ious[d].argmax() if len(ious[d]) > 0 else -1

        if best_iou >= iou_thresh and not gt_matched[best_g]:
            tp += 1
            gt_matched[best_g] = True
            matched_ious.append(best_iou)
        else:
            fp += 1

    fn = int((~gt_matched).sum())
    return tp, fp, fn, matched_ious


def compute_global_ap(gt_infos, det_annos, class_name, iou_thresh):
    """Compute AP using 11-point interpolation (PASCAL VOC style)."""
    all_det_boxes, all_det_scores, all_det_frame_ids = [], [], []
    frame_gt_boxes, frame_gt_matched = {}, {}
    total_gt = 0

    for i in range(len(gt_infos)):
        gt_anno = gt_infos[i]['annos']
        gt_mask = (gt_anno['name'] == class_name)
        gt_boxes = gt_anno['gt_boxes_lidar'][gt_mask] if 'gt_boxes_lidar' in gt_anno else np.zeros((0, 7))
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
    indices = np.argsort(-all_det_scores)
    all_det_boxes = all_det_boxes[indices]
    all_det_scores = all_det_scores[indices]
    all_det_frame_ids = all_det_frame_ids[indices]

    nd = len(all_det_boxes)
    tp, fp = np.zeros(nd), np.zeros(nd)

    for d in range(nd):
        frame_id = all_det_frame_ids[d]
        det_box = all_det_boxes[d:d + 1]
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

    tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
    rec = tp_cum / total_gt
    prec = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(np.float64).eps)

    # 11-point interpolation
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        p = np.max(prec[rec >= t]) if np.sum(rec >= t) > 0 else 0
        ap += p / 11.0
    return ap


def compute_full_metrics(gt_infos, det_annos, class_name):
    """Compute AP@0.3/0.5/0.7, average IoU, precision, recall, F1."""
    results = {}

    # AP at different thresholds
    for iou_thresh in [0.3, 0.5, 0.7]:
        results[f'AP@{iou_thresh}'] = compute_global_ap(gt_infos, det_annos, class_name, iou_thresh) * 100

    # Per-frame stats at IoU=0.5 for precision/recall/F1/avg_iou
    total_tp, total_fp, total_fn = 0, 0, 0
    all_ious = []

    for i in range(len(gt_infos)):
        gt_anno = gt_infos[i]['annos']
        gt_mask = (gt_anno['name'] == class_name)
        gt_boxes = gt_anno['gt_boxes_lidar'][gt_mask] if 'gt_boxes_lidar' in gt_anno else np.zeros((0, 7))

        det = det_annos[i]
        det_mask = (det['name'] == class_name)
        det_boxes = det['boxes_lidar'][det_mask]
        det_scores = det['score'][det_mask]

        tp, fp, fn, matched_ious = compute_per_frame_stats(gt_boxes, det_boxes, det_scores, 0.5)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        all_ious.extend(matched_ious)

    precision = total_tp / max(total_tp + total_fp, 1) * 100
    recall = total_tp / max(total_tp + total_fn, 1) * 100
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    avg_iou = np.mean(all_ious) * 100 if all_ious else 0.0

    results['Precision'] = precision
    results['Recall'] = recall
    results['F1'] = f1
    results['Avg IoU'] = avg_iou
    results['TP'] = total_tp
    results['FP'] = total_fp
    results['FN'] = total_fn

    return results


# ============================================================
# Visualization
# ============================================================

def plot_ablation(all_results, save_path='ablation_results.png'):
    """Generate publication-quality ablation charts."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    names = list(all_results.keys())
    short_names = []
    for n in names:
        if 'Source' in n:
            short_names.append('Source-Only\n(Baseline)')
        elif 'Few' in n:
            short_names.append('Few-Shot\n(10% GT)')
        elif 'Ours' in n or 'MixUp' in n or 'Stage' in n:
            short_names.append('Ours\n(MS3D+SSDA3D)')
        else:
            short_names.append(n[:20])

    colors = ['#B4B2A9', '#378ADD', '#D85A30']  # gray, blue, coral
    if len(names) > 3:
        colors = plt.cm.Set2(np.linspace(0, 1, len(names)))

    # --- Chart 1: AP comparison ---
    ax = axes[0]
    x = np.arange(len(names))
    width = 0.25
    for j, iou in enumerate([0.3, 0.5, 0.7]):
        vals = [all_results[n].get(f'AP@{iou}', 0) for n in names]
        bars = ax.bar(x + j * width, vals, width, label=f'AP@{iou}',
                      color=[c for c in colors], alpha=0.5 + j * 0.2,
                      edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xlabel('')
    ax.set_ylabel('AP (%)', fontsize=12)
    ax.set_title('3D Average Precision (Pedestrian)', fontsize=13, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(short_names, fontsize=9)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # --- Chart 2: Precision / Recall / F1 ---
    ax = axes[1]
    metrics = ['Precision', 'Recall', 'F1']
    x = np.arange(len(names))
    width = 0.25
    for j, m in enumerate(metrics):
        vals = [all_results[n].get(m, 0) for n in names]
        bars = ax.bar(x + j * width, vals, width, label=m,
                      edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Precision / Recall / F1 @ IoU=0.5', fontsize=13, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(short_names, fontsize=9)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # --- Chart 3: Average IoU comparison ---
    ax = axes[2]
    iou_vals = [all_results[n].get('Avg IoU', 0) for n in names]
    bars = ax.bar(short_names, iou_vals, color=colors[:len(names)],
                  edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, iou_vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Previous team reference line
    ax.axhline(y=53.2, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax.text(len(names) - 0.5, 54.5, "Previous team's IoU (53.2%)",
            fontsize=8, color='red', ha='right', style='italic')

    ax.set_ylabel('Average IoU (%)', fontsize=12)
    ax.set_title('Average IoU of Matched Detections', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"\n📊 Chart saved: {save_path}")
    plt.close()


def print_summary_table(all_results):
    """Print a formatted summary table."""
    print("\n" + "=" * 95)
    print("  ABLATION STUDY: PEDESTRIAN 3D DETECTION")
    print("=" * 95)

    header = f"{'Experiment':<45} {'AP@0.3':>7} {'AP@0.5':>7} {'AP@0.7':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'IoU':>7}"
    print(header)
    print("-" * 95)

    for name, r in all_results.items():
        row = (f"{name:<45} "
               f"{r.get('AP@0.3', 0):>6.2f}% "
               f"{r.get('AP@0.5', 0):>6.2f}% "
               f"{r.get('AP@0.7', 0):>6.2f}% "
               f"{r.get('Precision', 0):>6.2f}% "
               f"{r.get('Recall', 0):>6.2f}% "
               f"{r.get('F1', 0):>6.2f}% "
               f"{r.get('Avg IoU', 0):>6.2f}%")
        print(row)

    print("-" * 95)
    print(f"  {'Previous Team (reference):':<45} {'---':>7} {'---':>7} {'---':>7} {'---':>7} {'---':>7} {'---':>7} {'53.20%':>7}")
    print("=" * 95)

    # Detection counts
    print(f"\n{'Experiment':<45} {'TP':>6} {'FP':>6} {'FN':>6}")
    print("-" * 65)
    for name, r in all_results.items():
        print(f"{name:<45} {r.get('TP', 0):>6} {r.get('FP', 0):>6} {r.get('FN', 0):>6}")
    print()


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("  EVALUATION METRICS")
    print("=" * 80)
    print("  AP (Average Precision): The gold standard in 3D detection.")
    print("  Computed at IoU thresholds 0.3, 0.5, 0.7 using 11-point interpolation.")
    print("  Unlike simple average IoU, AP penalizes both missed detections and false positives.")
    print("=" * 80)

    # Load ground truth
    gt_path = '../data/custom/custom_infos_val.pkl'
    if not os.path.exists(gt_path):
        print(f"  Ground truth not found: {gt_path}")
        sys.exit(1)

    with open(gt_path, 'rb') as f:
        gt_infos = pickle.load(f)
    print(f"\n  Loaded {len(gt_infos)} val frames")

    # Define experiments — modify paths as needed
    experiments = OrderedDict({
        "Source-Only (Pretrained PV-RCNN)":
            "../output/custom_models/pv_rcnn/default/eval/epoch_8369/val/default/result.pkl",
        "Few-Shot (10% GT Fine-tune)":
            "../output/custom_models/pv_rcnn/default/eval/epoch_80/val/default/result.pkl",
        "Ours (MS3D + SSDA3D MixUp)":
            "../output/stage2_mixup/centerpoint_custom_mixup/debug_dummy_run/eval/eval_with_train/epoch_2/val/result.pkl",
    })

    # Evaluate each experiment
    all_results = OrderedDict()

    for exp_name, res_file in experiments.items():
        if not os.path.exists(res_file):
            print(f"  [SKIP] {exp_name}: {res_file} not found")
            continue

        print(f"  Evaluating: {exp_name}...")
        with open(res_file, 'rb') as f:
            det_annos = pickle.load(f)

        results = compute_full_metrics(gt_infos, det_annos, 'Pedestrian')
        all_results[exp_name] = results

    if not all_results:
        print("\n  No experiments found. Check result.pkl paths.")
        sys.exit(1)

    # Print table
    print_summary_table(all_results)

    # Generate chart
    save_path = '../output/ablation_results.png'
    plot_ablation(all_results, save_path=save_path)

    print("Done! Open the chart to see the visual comparison.")