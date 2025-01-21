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


class FloorPlanDrawer:
    def __init__(
        self,
        floor_plan: MatLike,
        history_length: int = 10,
        transform_matrix: np.ndarray | None = None,
    ) -> None:
        self.viz = floor_plan.copy()
        self.history_length = history_length
        self.transform_matrix = transform_matrix

        self.history: dict[int, dict[int, deque[np.ndarray]]] = {}

    def draw(
        self,
        projections: dict[int, list[tuple[TrackedObject, np.ndarray]]],
    ) -> MatLike:
        viz = self.viz.copy()

        for cid, tracked_objects in projections.items():
            for obj, point in tracked_objects:
                color = Palette.choose_color(obj.id)

                if self.transform_matrix is not None:
                    point = (
                        (self.transform_matrix @ np.hstack((point, 1)).reshape(-1, 1))
                        .flatten()
                        .astype(int)
                    )

                self.history.setdefault(cid, {}).setdefault(
                    obj.id, deque(maxlen=self.history_length)
                ).append(point)

                for i in range(1, len(self.history[cid][obj.id])):
                    cv2.line(
                        viz,
                        tuple(self.history[cid][obj.id][i - 1]),
                        tuple(self.history[cid][obj.id][i]),
                        color=color,
                        thickness=2,
                    )

                cv2.circle(
                    viz,
                    tuple(point),
                    radius=5,
                    color=color,
                    thickness=-1,
                )

                label = f"{cid} {obj.id}"
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
                    viz,
                    label,
                    (text_x, text_y),
                    font,
                    font_scale,
                    color,
                    font_thickness,
                )

        return viz


class GlobalFloorPlanDrawer:
    def __init__(
        self,
        floor_plan: MatLike,
        history_length: int = 10,
        transform_matrix: np.ndarray | None = None,
    ) -> None:
        self.viz = floor_plan.copy()
        self.history_length = history_length
        self.transform_matrix = transform_matrix

        self.history: dict[int, deque[np.ndarray]] = {}

    def draw(
        self,
        objects: list[GlobalObject],
    ) -> MatLike:
        viz = self.viz.copy()

        for obj in objects:
            point = obj.position
            color = Palette.choose_color(obj.id)

            if self.transform_matrix is not None:
                point = (
                    (self.transform_matrix @ np.hstack((point, 1)).reshape(-1, 1))
                    .flatten()
                    .astype(int)
                )

            self.history.setdefault(obj.id, deque(maxlen=self.history_length)).append(
                point
            )

            for i in range(1, len(self.history[obj.id])):
                cv2.line(
                    viz,
                    tuple(self.history[obj.id][i - 1]),
                    tuple(self.history[obj.id][i]),
                    color=color,
                    thickness=2,
                )

            cv2.circle(
                viz,
                tuple(point),
                radius=5,
                color=color,
                thickness=-1,
            )

            label = f"{obj.id}"
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
                viz,
                label,
                (text_x, text_y),
                font,
                font_scale,
                color,
                font_thickness,
            )

        return viz
