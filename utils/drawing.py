from collections import deque

import cv2
import numpy as np
import supervision as sv
from cv2.typing import MatLike
from norfair import Palette, draw_points
from norfair.tracker import TrackedObject

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
    draw_points(annot_frame, tracked, radius=8, text_size=1)
    return annot_frame


def draw_floor_plan(
    floor_plan: MatLike,
    projections: dict[int, tuple[TrackedObject, list[tuple[int, np.ndarray]]]],
    transform_matrix: np.ndarray,
) -> MatLike:
    viz = floor_plan.copy()
    for cid, (_, projected_points) in projections.items():
        for id, point in projected_points:
            transformed_point = point.astype(np.int32).reshape(-1, 1)
            new_point = transform_matrix @ transformed_point
            new_x, new_y = int(new_point[0][0]), int(new_point[1][0])
            color = Palette.choose_color(id)
            cv2.circle(viz, (new_x, new_y), radius=5, color=color, thickness=-1)
            label = f"{cid} {id}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(
                label, font, font_scale, font_thickness
            )
            text_x = new_x - text_width // 2
            text_y = new_y - 10
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

        self.history: dict[int, list[np.ndarray]] = {}

    def draw(self, tracked_objects: list[TrackedObject]) -> MatLike:
        viz = self.viz.copy()

        for obj in tracked_objects:
            color = Palette.choose_color(obj.id)

            point = np.mean(np.array(obj.estimate), axis=0)
            if self.transform_matrix is not None:
                point = (
                    (self.transform_matrix @ np.hstack((point, 1)).reshape(-1, 1))
                    .flatten()
                    .astype(int)
                )

            self.history.setdefault(
                obj.id,
                deque(maxlen=self.history_length),
            ).append(point)

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
