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

    cv2.putText(
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
        self, projections: dict[int, list[tuple[TrackedObject, np.ndarray]]]
    ) -> MatLike:
        viz = self.floor_plan.copy()
        for cid, tracked_objects in projections.items():
            for obj, point in tracked_objects:
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
            self._draw_point(viz, point, color, f"{obj.id}")
        return viz
