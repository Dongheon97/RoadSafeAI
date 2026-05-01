import argparse
import pickle
from pathlib import Path

import numpy as np


PED_CLASS_ID = 2


def filter_frame(frame_entry):
    gt_boxes = frame_entry.get('gt_boxes')
    if gt_boxes is None:
        return frame_entry

    gt_boxes = np.asarray(gt_boxes)
    if gt_boxes.size == 0:
        return frame_entry

    mask = np.abs(gt_boxes[:, 7].astype(np.int32)) == PED_CLASS_ID

    filtered = dict(frame_entry)
    filtered['gt_boxes'] = gt_boxes[mask]

    for key in ('num_pts', 'memory_counter'):
        if key in filtered and filtered[key] is not None:
            filtered[key] = np.asarray(filtered[key])[mask]

    return filtered


def main():
    parser = argparse.ArgumentParser(description='Filter final pseudo labels to Pedestrian-only entries.')
    parser.add_argument('--input', required=True, help='Path to input final_ps_dict.pkl')
    parser.add_argument('--output', required=True, help='Path to filtered pedestrian-only pseudo label pickle')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open('rb') as f:
        ps_dict = pickle.load(f)

    filtered_dict = {frame_id: filter_frame(entry) for frame_id, entry in ps_dict.items()}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('wb') as f:
        pickle.dump(filtered_dict, f)

    total_frames = len(filtered_dict)
    total_boxes = sum(entry['gt_boxes'].shape[0] for entry in filtered_dict.values())
    print(f'Wrote {output_path}')
    print(f'frames={total_frames} pedestrian_boxes={total_boxes}')


if __name__ == '__main__':
    main()
