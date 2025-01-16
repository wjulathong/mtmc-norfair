import time
from pathlib import Path

import cv2
from cv2.typing import MatLike


class FPS:
    def __init__(self) -> None:
        self.prev_time: float = time.time()
        self.fps: float = 0

    def update(self):
        cur_time = time.time()
        self.fps = 1 / (cur_time - self.prev_time)
        self.prev_time = cur_time


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


# import enum
# import queue
# import multiprocessing as mp

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
