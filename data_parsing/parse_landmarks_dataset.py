import os
from typing import List

from utils.data_utils import LandmarksAnnotation

class LandmarksDatasetParser:
    def __init__(self, db_root: str) -> None:
        self.db_root = db_root
    
    def parse(self) -> List[LandmarksAnnotation]:
        annotations = []

        for sample_dir in os.listdir(self.db_root):
            if not os.path.isdir(os.path.join(self.db_root, sample_dir)):
                continue
            
            image_path = os.path.join(self.db_root, sample_dir, f"{sample_dir}.jpg")
            landmarks3d_path = os.path.join(self.db_root, sample_dir, f"{sample_dir}_pts3d.npy")
            landmarks2d_path = os.path.join(self.db_root, sample_dir, f"{sample_dir}_pts2d.npy")

            annotation = LandmarksAnnotation(
                image_path=image_path,
                landmarks3d_path=landmarks3d_path,
                landmarks2d_path=landmarks2d_path
            )
            
            annotations.append(annotation)
        
        return annotations