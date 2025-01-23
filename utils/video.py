import time
from pathlib import Path

import cv2
from cv2.typing import MatLike


class FPS:
    def __init__(self) -> None:
        self.prev_time: float = time.perf_counter()
        self.fps: float = 0

    def update(self):
        cur_time = time.perf_counter()
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
