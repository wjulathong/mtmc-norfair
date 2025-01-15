import json
from pathlib import Path

import cv2
import norfair
import numpy as np
from cv2.typing import MatLike
from norfair.tracker import TrackedObject
from norfair.drawing.path import Paths

# import enum
# import time
# import queue
# import multiprocessing as mp


def preview_frame(frame: MatLike, sf: float) -> MatLike:
    return cv2.resize(frame, dsize=None, fx=sf, fy=sf, interpolation=cv2.INTER_AREA)


class FrameGetter:
    def __init__(self, video_path: Path, target_fps: float | None = None) -> None:
        self.cap = cv2.VideoCapture(str(video_path))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.target_fps = target_fps or self.fps

        if self.target_fps <= 0 or self.target_fps > self.fps:
            raise ValueError(
                "Target FPS must be greater than 0 and less than or equal to the video's original FPS."
            )

        self.frames_to_skip = int(self.fps / self.target_fps) - 1

    def read(self) -> MatLike | None:
        ret, frame = self.cap.read()
        if not ret:
            return None

        for _ in range(self.frames_to_skip):
            self.cap.read()

        return frame

    def seek_forward(self, sec: int):
        current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        frames_to_skip = int(self.fps * sec)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame + frames_to_skip)

    def seek_backward(self, sec: int):
        current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        frames_to_skip = int(self.fps * sec)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current_frame - frames_to_skip))

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()


def load_homography(calibration_path: Path) -> np.ndarray:
    with open(calibration_path) as f:
        data = json.load(f)
    return np.array(data["homography_matrix"])


def project_points(
    tracked: list[TrackedObject], homography: np.ndarray
) -> list[tuple[int, np.ndarray]]:
    objects = [(obj.id, np.mean(np.array(obj.estimate), axis=0)) for obj in tracked]
    if not objects:
        return []

    ids, points = zip(*objects)
    points_array = np.array(points, dtype=np.float32)
    homogeneous_points = np.column_stack([points_array, np.ones(len(points_array))])
    projected_points = homography @ homogeneous_points.T
    projected_points /= projected_points[2]
    return list(zip(ids, projected_points.T))


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
            color = norfair.Palette.choose_color(id)
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


def draw_global_floor_plan(
    floor_plan: MatLike,
    global_tracked: TrackedObject,
    transform_matrix: np.ndarray,
) -> MatLike:
    viz = floor_plan.copy()

    drawer = Paths()
    viz = drawer.draw(viz, global_tracked)

    # for t_obj in global_tracked:
    #     point = np.append(t_obj.last_detection.points[0], 1)
    #     transformed_point = point.astype(np.int32).reshape(-1, 1)
    #     new_point = transform_matrix @ transformed_point
    #     new_x, new_y = int(new_point[0][0]), int(new_point[1][0])
    #     color = norfair.Palette.choose_color(t_obj.id)
    #     cv2.circle(viz, (new_x, new_y), radius=5, color=color, thickness=-1)
    #     label = f"{t_obj.id}"
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     font_scale = 1
    #     font_thickness = 2
    #     (text_width, text_height), _ = cv2.getTextSize(
    #         label, font, font_scale, font_thickness
    #     )
    #     text_x = new_x - text_width // 2
    #     text_y = new_y - 10
    #     cv2.putText(
    #         viz,
    #         label,
    #         (text_x, text_y),
    #         font,
    #         font_scale,
    #         (0, 0, 0),
    #         font_thickness + 1,
    #     )
    #     cv2.putText(
    #         viz,
    #         label,
    #         (text_x, text_y),
    #         font,
    #         font_scale,
    #         color,
    #         font_thickness,
    #     )
    return viz

    # for cid, (_, projected_points) in projections.items():
    #     for id, point in projected_points:
    #         transformed_point = point.astype(np.int32).reshape(-1, 1)
    #         new_point = transform_matrix @ transformed_point
    #         new_x, new_y = int(new_point[0][0]), int(new_point[1][0])
    #         color = norfair.Palette.choose_color(id)
    #         cv2.circle(viz, (new_x, new_y), radius=5, color=color, thickness=-1)
    #         label = f"{cid} {id}"
    #         font = cv2.FONT_HERSHEY_SIMPLEX
    #         font_scale = 1
    #         font_thickness = 2
    #         (text_width, text_height), _ = cv2.getTextSize(
    #             label, font, font_scale, font_thickness
    #         )
    #         text_x = new_x - text_width // 2
    #         text_y = new_y - 10
    #         cv2.putText(
    #             viz,
    #             label,
    #             (text_x, text_y),
    #             font,
    #             font_scale,
    #             (0, 0, 0),
    #             font_thickness + 1,
    #         )
    #         cv2.putText(
    #             viz,
    #             label,
    #             (text_x, text_y),
    #             font,
    #             font_scale,
    #             color,
    #             font_thickness,
    #         )
    # return viz


# class SeekEvent(enum.Enum):
#     FORWARD = enum.auto()
#     BACKWARD = enum.auto()
#
#
# class ReaderState(enum.Enum):
#     READY = enum.auto()
#     NO_NEW_FRAME = enum.auto()
#     END_OF_STREAM = enum.auto()
#     ERROR = enum.auto()
#
#
# class MPFrameGetter:
#     def __init__(
#         self,
#         video_path: Path,
#         target_fps: float | None = None,
#         max_queue_size: int = 32,
#     ) -> None:
#         self.frame_queue: mp.Queue[MatLike] = mp.Queue(maxsize=max_queue_size)
#         self.control_event = mp.Event()
#         self.seek_queue: mp.Queue[tuple[SeekEvent, int]] = mp.Queue()
#
#         cap = cv2.VideoCapture(str(video_path))
#         self.source_fps = cap.get(cv2.CAP_PROP_FPS)
#         cap.release()
#
#         valid_target_fps = not (target_fps is None or target_fps > self.source_fps)
#         self.target_fps = target_fps if valid_target_fps else self.source_fps
#
#         self.frame_interval = 1.0 / self.target_fps
#
#         self.state = ReaderState.READY
#         self.last_frame_time = 0
#
#         self.reader_process = mp.Process(
#             target=self._reader_worker,
#             args=(
#                 str(video_path),
#                 self.frame_queue,
#                 self.control_event,
#                 self.seek_queue,
#                 self.source_fps,
#                 self.target_fps,
#             ),
#         )
#         self.reader_process.daemon = True
#         self.reader_process.start()
#
#     def _reader_worker(
#         video_path: str,
#         frame_queue: mp.Queue[MatLike],
#         control_event: mp.Event,
#         seek_queue: mp.Queue[tuple[SeekEvent, int]],
#         source_fps: float,
#         target_fps: float,
#     ) -> None:
#         cap = cv2.VideoCapture(video_path)
#         frame_interval = 1.0 / target_fps
#         frames_to_skip = max(0, int(source_fps / target_fps) - 1)
#         last_frame_time = time.time()
#
#         try:
#             while not control_event.is_set():
#                 try:
#                     direction, seconds = seek_queue.get_nowait()
#                     current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
#                 except queue.Empty:
#                     pass
#         finally:
#             pass
