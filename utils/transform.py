import json
from pathlib import Path

import numpy as np
from cv2.typing import MatLike
from norfair.tracker import TrackedObject


def load_homography(calibration_path: Path) -> np.ndarray:
    with open(calibration_path) as f:
        data = json.load(f)
    return np.array(data["homography_matrix"])


def project_points(
    tracked_objects: list[TrackedObject], homography: np.ndarray
) -> list[tuple[TrackedObject, np.ndarray]]:
    if not tracked_objects:
        return []

    points_array = np.array(
        [np.mean(np.array(obj.estimate), axis=0) for obj in tracked_objects],
        dtype=np.float32,
    )
    homogeneous_points = np.column_stack([points_array, np.ones(len(points_array))])
    projected_points = homography @ homogeneous_points.T
    projected_points /= projected_points[2]
    projected_points = projected_points[:2].T
    return list(zip(tracked_objects, projected_points))


def prepare_floor_plan(floor_plan: MatLike, scale: float) -> tuple[MatLike, np.ndarray]:
    height, width = floor_plan.shape[:2]
    new_width = int(width * scale)
    new_height = int(height * scale)

    pad_x = (new_width - width) // 2
    pad_y = (new_height - height) // 2

    scaled_plan = np.full((new_height, new_width, 3), (255, 255, 255), dtype=np.uint8)
    scaled_plan[pad_y : pad_y + height, pad_x : pad_x + width] = floor_plan

    transform_matrix = np.array(
        [
            [1, 0, pad_x],
            [0, 1, pad_y],
        ],
        dtype=np.float32,
    )
    return scaled_plan, transform_matrix
