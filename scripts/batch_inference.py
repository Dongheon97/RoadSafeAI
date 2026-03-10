import argparse
import glob
from pathlib import Path
import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

# Define a dedicated Dataset for reading custom data
class BatchInferenceDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.npy'):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = Path(root_path)
        self.ext = ext
        data_file_list = glob.glob(str(self.root_path / f'*{self.ext}')) if self.root_path.is_dir() else [str(self.root_path)]
        # Extract numbers from the filename and convert to integers for numerical sorting
        data_file_list.sort(key=lambda f: int(''.join(filter(str.isdigit, Path(f).stem)) or 0))
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': Path(self.sample_file_list[index]).stem,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

def parse_config():
    parser = argparse.ArgumentParser(description='Batch Inference Script')
    parser.add_argument('--cfg_file', type=str, required=True, help='specify the config for inference')
    parser.add_argument('--data_path', type=str, required=True, help='specify the point cloud data directory')
    parser.add_argument('--ckpt', type=str, required=True, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.npy', help='specify the extension of your point cloud data file')
    parser.add_argument('--output_dir', type=str, default='../data/custom/labels_predicted', help='directory to save predicted labels')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    return args, cfg

def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('--- Start Batch Inference ---')
    
    # Initialize dataset
    inference_dataset = BatchInferenceDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=args.data_path, ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples to process: \t{len(inference_dataset)}')

    # Build and load model
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=inference_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f'Labels will be saved to: {output_dir.resolve()}')

    # Start inference
    with torch.no_grad():
        for idx, data_dict in enumerate(inference_dataset):
            frame_id = data_dict['frame_id']
            logger.info(f'Processing sample index: {idx + 1} (Frame: {frame_id})')
            
            data_dict = inference_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            # Extract prediction results
            pred_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
            pred_labels = pred_dicts[0]['pred_labels'].cpu().numpy()
            pred_scores = pred_dicts[0]['pred_scores'].cpu().numpy()

            # Write to txt file
            label_path = output_dir / f"{frame_id}.txt"
            with open(label_path, 'w') as f:
                for i in range(len(pred_boxes)):
                    box = pred_boxes[i]
                    cls_name = cfg.CLASS_NAMES[pred_labels[i] - 1]
                    score = pred_scores[i]
                    # Format: Class x y z dx dy dz heading score
                    f.write(f"{cls_name} {box[0]:.4f} {box[1]:.4f} {box[2]:.4f} {box[3]:.4f} {box[4]:.4f} {box[5]:.4f} {box[6]:.4f} {score:.4f}\n")

    logger.info('--- Batch Inference Completed ---')

if __name__ == '__main__':
    main()