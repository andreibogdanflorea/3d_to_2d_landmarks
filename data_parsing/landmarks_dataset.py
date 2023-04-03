import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

from data_parsing.parse_landmarks_dataset import LandmarksDatasetParser

class LandmarksDataset(Dataset):
    """
    Read dataset for training landmarks mapping
    """

    def __init__(self, db_root: str, return_image: bool = False) -> None:
        super().__init__()
        self.db_root = db_root
        self.return_image = return_image

        self.db_parser = LandmarksDatasetParser(db_root)
        self.annotations = self.db_parser.parse()
    
    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int) -> list([torch.Tensor, torch.Tensor]):
        annotation = self.annotations[index]

        landmarks3d = np.load(annotation.landmarks3d_path)
        landmarks2d = np.load(annotation.landmarks2d_path)

        landmarks3d = torch.from_numpy(landmarks3d).to(torch.float32).reshape(-1)
        landmarks2d = torch.from_numpy(landmarks2d).to(torch.float32).reshape(-1)

        if self.return_image:
            image = cv2.imread(annotation.image_path)
            image = torch.from_numpy(image).to(torch.uint8)

            return landmarks3d, landmarks2d, image
        
        return landmarks3d, landmarks2d