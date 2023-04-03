import argparse
from yaml import safe_load
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2

from utils.model_utils import load_mapping_network
from utils.plotting_utils import plot_points_on_image
from data_parsing.landmarks_dataset import LandmarksDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Generate samples from the generator")
    parser.add_argument(
        "--config", help="training configuration file", required=True, type=str
    )

    return parser.parse_args()

class Tester:
    def __init__(self, config_file: str) -> None:
        self.config_file = config_file
        self.config = self.parse_config(config_file)

        self.model = load_mapping_network(self.config).to('cuda')

        self.test_db_path = self.config["DATASET_INFERENCE"]["PATH"]

    def parse_config(self, config_file: str) -> dict:
        config = {}
        if os.path.isfile(config_file) and config_file.endswith(".yml"):
            with open(config_file, "r") as f_config:
                config = safe_load(f_config) 
        else:
            print("Invalid config path: {}".format(config_file))
        
        return config

    def make_inferences(self):
        self.model.eval()
        with torch.no_grad():
            for sample_dir in os.listdir(self.test_db_path):
                sample_dir_path = os.path.join(self.test_db_path, sample_dir)
                if not os.path.isdir(sample_dir_path):
                    continue
                
                image_path = os.path.join(sample_dir_path, f'{sample_dir}.jpg')
                landmarks3d_path = image_path.replace('.jpg', '_pts3d.npy')

                image = cv2.imread(image_path)
                landmarks3d = np.load(landmarks3d_path)
                landmarks3d = torch.tensor(landmarks3d).reshape(-1).unsqueeze(0)

                landmarks3d = landmarks3d.cuda(non_blocking=True)
                preds = self.model(landmarks3d)

                pred_points2d = preds[0].detach().cpu().numpy()
                input_points3d = landmarks3d[0].detach().cpu().numpy()
                image_with_points = plot_points_on_image(image, pred_points2d, input_points3d)
                image_with_points = cv2.cvtColor(image_with_points, cv2.COLOR_RGB2BGR)

                image_save_path = image_path.replace('.jpg', '_draw_pts.jpg')
                cv2.imwrite(image_save_path, image_with_points)

                preds_save_path = image_path.replace('.jpg', '_preds2d.npy')
                with open(preds_save_path, 'wb') as f_preds:
                    np.save(f_preds, pred_points2d)

if __name__ == "__main__":
    args = parse_args()
    tester = Tester(config_file=args.config)
    tester.make_inferences()
