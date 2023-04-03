from dataclasses import dataclass, field

@dataclass
class LandmarksAnnotation:
    """
    Definition of a class that encompasses paths to an image and its 2d and 3d landmark annotations
    """
    
    image_path: str = field(default="")
    landmarks2d_path: str = field(default="")
    landmarks3d_path: str = field(default="")


