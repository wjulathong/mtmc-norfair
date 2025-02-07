from collections import deque

import cv2
import numpy as np
import supervision as sv
from cv2.typing import MatLike
from norfair import Palette, draw_points
from norfair.tracker import TrackedObject

from utils.global_association import GlobalObject

BBOX_ANN = sv.BoxAnnotator(thickness=3)
LABEL_ANN = sv.LabelAnnotator()


def preview_frame(frame: MatLike, sf: float) -> MatLike:
    return cv2.resize(frame, dsize=None, fx=sf, fy=sf, interpolation=cv2.INTER_AREA)


def resize_fixed_width(img: MatLike, fixed_width: int):
    height, width = img.shape[:2]
    scale = fixed_width / width
    new_height = int(height * scale)
    return cv2.resize(img, (fixed_width, new_height), interpolation=cv2.INTER_AREA)


def resize_fixed_height(img: MatLike, fixed_height: int):
    height, width = img.shape[:2]
    scale = fixed_height / height
    new_width = int(width * scale)
    return cv2.resize(img, (new_width, fixed_height), interpolation=cv2.INTER_AREA)


def resize_fixed_dimensions(img: MatLike, shape: tuple[int, int]):
    ori_shape = np.array(img.shape[:2])
    target_shape = np.array(shape)[::-1]
    scale = np.min(target_shape / ori_shape)
    new_shape = tuple((ori_shape * scale)[::-1].astype(int))
    return cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)


def debug_detections(cropped_images: list[MatLike]) -> MatLike:
    target_height = max(img.shape[0] for img in cropped_images)
    min_width = min(img.shape[1] for img in cropped_images)
    font_scale = 1.0 * (min_width / 150)
    thickness = max(2, int(font_scale * 4))
    resized_imgs = [
        draw_text_on_frame(
            resize_fixed_height(img, target_height),
            f"{i}",
            "top-left",
            font_scale=font_scale,
            thickness=thickness,
        )
        for i, img in enumerate(cropped_images)
    ]
    return cv2.hconcat(resized_imgs)


def debug_camera(imgs: list[tuple[int, MatLike]]) -> MatLike:
    target_width = max(img.shape[1] for _, img in imgs)
    font_scale = 1.0 * (target_width / 1000)
    thickness = max(2, int(font_scale * 4))
    resized_imgs = [
        draw_text_on_frame(
            resize_fixed_width(img, target_width),
            f"{i}",
            "bottom-right",
            font_scale=font_scale,
            color=(0, 255, 0),
            thickness=thickness,
        )
        for i, img in imgs
    ]
    return cv2.vconcat(resized_imgs)


def draw_text_on_frame(
    frame: MatLike,
    text: str,
    position: str = "bottom-left",
    font_scale: float = 1.0,
    color: tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
):
    height, width = frame.shape[:2]
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    if position == "top-left":
        text_size = cv2.getTextSize(text, font_face, font_scale, thickness)[0]
        coordinate = (10, 10 + text_size[1])
    elif position == "top-left":
        text_size = cv2.getTextSize(text, font_face, font_scale, thickness)[0]
        coordinate = (width - text_size[0] - 10, 10 + text_size[1])
    elif position == "bottom-left":
        coordinate = (10, height - 10)
    elif position == "bottom-right":
        text_size = cv2.getTextSize(text, font_face, font_scale, thickness)[0]
        coordinate = (width - text_size[0] - 10, height - 10)
    else:
        raise ValueError(
            "Invalid position. Choose from 'top-left', 'top-right', 'bottom-left', or 'bottom-right'."
        )

    return cv2.putText(
        frame,
        text,
        coordinate,
        font_face,
        font_scale,
        color,
        thickness,
    )


def annotate(
    frame: MatLike, detections: sv.Detections, tracked_objects: list[TrackedObject]
):
    labels = [
        f"{cname} {conf:.2f}"
        for cname, conf in zip(detections["class_name"], detections.confidence)
    ]
    annotated_frame = BBOX_ANN.annotate(scene=frame.copy(), detections=detections)
    annotated_frame = LABEL_ANN.annotate(
        scene=annotated_frame, detections=detections, labels=labels
    )
    draw_points(annotated_frame, tracked_objects, radius=8, text_size=1)
    return annotated_frame


class BaseFloorPlanDrawer:
    def __init__(
        self,
        floor_plan: MatLike,
        history_length: int = 10,
        transform_matrix: np.ndarray | None = None,
    ) -> None:
        self.floor_plan = floor_plan.copy()
        self.history_length = history_length
        self.transform_matrix = transform_matrix
        self.history = {}

    def _transform_point(self, point: np.ndarray) -> np.ndarray:
        if self.transform_matrix is not None:
            homogeneous = np.array([point[0], point[1], 1])
            transformed = self.transform_matrix @ homogeneous
            point = transformed.astype(int)
        return point

    def _draw_history(
        self,
        viz: MatLike,
        point_history: deque[np.ndarray],
        color: tuple[int, int, int],
    ):
        for i in range(1, len(point_history)):
            cv2.line(
                viz,
                tuple(point_history[i - 1]),
                tuple(point_history[i]),
                color=color,
                thickness=2,
            )

    def _draw_point(
        self, viz: MatLike, point: np.ndarray, color: tuple[int, int, int], label: str
    ):
        cv2.circle(viz, tuple(point), radius=5, color=color, thickness=-1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )
        text_x = point[0] - text_width // 2
        text_y = point[1] - 10
        cv2.putText(
            viz,
            label,
            (text_x, text_y),
            font,
            font_scale,
            (0, 0, 0),
            font_thickness + 1,
        )
        cv2.putText(
            viz, label, (text_x, text_y), font, font_scale, color, font_thickness
        )


class FloorPlanDrawer(BaseFloorPlanDrawer):
    def draw(
        self, projections: dict[int, tuple[list[TrackedObject], list[np.ndarray]]]
    ) -> MatLike:
        viz = self.floor_plan.copy()
        for cid, (tracked_objects, projected_points) in projections.items():
            for obj, point in zip(tracked_objects, projected_points):
                color = Palette.choose_color(obj.id)
                point = self._transform_point(point)
                self.history.setdefault(cid, {}).setdefault(
                    obj.id, deque(maxlen=self.history_length)
                ).append(point)

                self._draw_history(viz, self.history[cid][obj.id], color)
                self._draw_point(viz, point, color, f"{cid} {obj.id}")
        return viz


class GlobalFloorPlanDrawer(BaseFloorPlanDrawer):
    def draw(self, objects: list[GlobalObject]) -> MatLike:
        viz = self.floor_plan.copy()
        for obj in objects:
            color = Palette.choose_color(obj.id)
            point = self._transform_point(obj.position)
            self.history.setdefault(obj.id, deque(maxlen=self.history_length)).append(
                point
            )

            self._draw_history(viz, self.history[obj.id], color)
            self._draw_point(
                viz, point, color, f"{obj.id}" if obj.is_active() else f"IA: {obj.id}"
            )
        return viz
