import numpy as np
import cv2

def plot_points_on_image(
    image: np.ndarray,
    points2d: np.ndarray,
    points3d: np.ndarray,
    points2d_color: tuple = (255, 0, 0),
    points3d_color: tuple = (0, 255, 0)
) -> np.ndarray:
    image_with_points = np.copy(image)
    image_with_points = cv2.cvtColor(image_with_points, cv2.COLOR_BGR2RGB)

    points3d = points3d.reshape(-1, 2)
    for x, y in points3d:
        center = (int(x), int(y))
        cv2.circle(image_with_points, center, 1, points3d_color, 1)

    points2d = points2d.reshape(-1, 2)
    for x, y in points2d:
        center = (int(x), int(y))
        cv2.circle(image_with_points, center, 1, points2d_color, 1)

    return image_with_points