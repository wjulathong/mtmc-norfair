import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import openvino as ov
import supervision as sv
from cv2.typing import MatLike
from norfair import Detection, OptimizedKalmanFilterFactory, Tracker
from norfair.tracker import TrackedObject
from scipy.spatial.distance import cosine
from ultralytics import YOLO

YOLO_MODEL_PATH = Path("./models/yolo11s.pt")
REID_MODEL_NAME = "person-reidentification-retail-0288"
REID_MODEL_SIZE = "FP16"
REID_MODEL_PATH = Path(
    f"./models/intel/{REID_MODEL_NAME}/{REID_MODEL_SIZE}/{REID_MODEL_NAME}.xml"
)


class Detector:
    def __init__(self, edge_margin: int = 20) -> None:
        self.model_path = YOLO_MODEL_PATH
        self.model = YOLO(self.model_path)
        self.edge_margin = edge_margin

    def detect(self, frame: MatLike) -> sv.Detections:
        results = self.model.predict(source=frame, conf=0.6, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[detections["class_name"] == "person"]
        detections = detections[
            (detections.xyxy[:, 0] >= self.edge_margin)
            & (detections.xyxy[:, 1] >= self.edge_margin)
            & (detections.xyxy[:, 2] <= frame.shape[1] - self.edge_margin)
            & (detections.xyxy[:, 3] <= frame.shape[0] - self.edge_margin)
        ]
        return detections


class PersonRecognizer:
    def __init__(self) -> None:
        self.core = ov.Core()
        self.model = self.core.read_model(REID_MODEL_PATH)
        self.model.reshape([1, 3, 256, 128])
        self.layout = ov.Layout("NCHW")
        self.compiled_model = self.core.compile_model(self.model)

        self.tensor_shape = self.compiled_model.input().shape
        self.tensor_width = self.tensor_shape[self.layout.get_index_by_name("width")]
        self.tensor_height = self.tensor_shape[self.layout.get_index_by_name("height")]

    def infer(self, mat: MatLike):
        tensor = self._create_input_tensor(mat)
        return self.compiled_model(tensor)[0]

    def _create_input_tensor(self, mat: MatLike):
        resized_mat = cv2.resize(mat, (self.tensor_width, self.tensor_height))
        return ov.Tensor(
            np.expand_dims(resized_mat.transpose(2, 0, 1).astype(np.float32), axis=0)
        )


def embedding_distance(
    matched_not_init_trackers: TrackedObject,
    unmatched_trackers: TrackedObject,
):
    snd_embeddings = []

    if unmatched_trackers.last_detection.embedding is not None:
        snd_embeddings.append(
            np.array(unmatched_trackers.last_detection.embedding).reshape(256)
        )

    for detection in unmatched_trackers.past_detections:
        if detection.embedding is not None:
            snd_embeddings.append(np.array(detection.embedding).reshape(256))

    if not snd_embeddings:
        return 2.0

    fst_embeddings = []
    for detection_fst in matched_not_init_trackers.past_detections:
        if detection_fst.embedding is not None:
            fst_embeddings.append(np.array(detection_fst.embedding).reshape(256))

    if not fst_embeddings:
        return 2.0

    min_distance = 2.0
    for snd_emb in snd_embeddings:
        for fst_emb in fst_embeddings:
            distance = cosine(snd_emb, fst_emb)
            min_distance = min(min_distance, distance)
    return min_distance


def mean_embedding_distance(
    matched_not_init_trackers: TrackedObject, unmatched_trackers: TrackedObject
):
    snd_embeddings = []
    if unmatched_trackers.last_detection.embedding is not None:
        snd_embeddings.append(
            np.array(unmatched_trackers.last_detection.embedding).reshape(256)
        )
    for detection in unmatched_trackers.past_detections:
        if detection.embedding is not None:
            snd_embeddings.append(np.array(detection.embedding).reshape(256))

    if not snd_embeddings:
        return 2.0

    snd_embedding = np.mean(snd_embeddings, axis=0)

    fst_embeddings = []
    for detection_fst in matched_not_init_trackers.past_detections:
        if detection_fst.embedding is not None:
            fst_embeddings.append(np.array(detection_fst.embedding).reshape(256))

    if not fst_embeddings:
        return 2.0

    fst_embedding = np.mean(fst_embeddings, axis=0)

    return cosine(snd_embedding, fst_embedding)


def embedding_distance_old(
    matched_not_init_trackers: TrackedObject,
    unmatched_trackers: TrackedObject,
):
    snd_embedding = unmatched_trackers.last_detection.embedding

    if snd_embedding is None:
        for detection in reversed(unmatched_trackers.past_detections):
            if detection.embedding is not None:
                snd_embedding = detection.embedding
                break
        else:
            return 2.0

    snd_embedding = np.array(snd_embedding).reshape(256)

    for detection_fst in matched_not_init_trackers.past_detections:
        if detection_fst.embedding is None:
            continue

        fst_embedding = np.array(detection_fst.embedding).reshape(256)

        dist = cosine(snd_embedding, fst_embedding)
        return dist
    return 2.0


class PersonTracker:
    def __init__(
        self,
        distance_function: str | Callable[[Detection, TrackedObject], float],
        reid_distance_function: Callable[["TrackedObject", "TrackedObject"], float],
        hit_counter_max: int = 10,
        distance_threshold: float = 50.0,
        reid_hit_counter_max: int = 300,
        reid_distance_threshold: float = 0.7,
        past_detections_length: int = 10,
        initialization_delay: int | None = None,
    ) -> None:
        self.tracker = Tracker(
            filter_factory=OptimizedKalmanFilterFactory(),
            distance_function=distance_function,
            hit_counter_max=hit_counter_max,
            distance_threshold=distance_threshold,
            reid_distance_function=reid_distance_function,
            reid_distance_threshold=reid_distance_threshold,
            reid_hit_counter_max=reid_hit_counter_max,
            past_detections_length=past_detections_length,
            initialization_delay=initialization_delay,
        )

    def update(self, detections: sv.Detections, reids: list[np.ndarray]):
        nf_detections = [
            Detection(
                points=np.array([(xyxy[0] + xyxy[2]) / 2, xyxy[3]]),
                scores=np.array([conf]),
                embedding=emb,
            )
            for xyxy, conf, emb in zip(detections.xyxy, detections.confidence, reids)
        ]
        return self.tracker.update(nf_detections)


class GlobalTracker:
    def __init__(
        self,
        total_tracker: int,
        distance_function: str | Callable[[Detection, TrackedObject], float],
        reid_distance_function: Callable[["TrackedObject", "TrackedObject"], float]
        | None = None,
        hit_counter_max: int = 10,
        distance_threshold: float = 50.0,
        reid_hit_counter_max: int = 300,
        reid_distance_threshold: float = 0.7,
        past_detections_length: int = 10,
        initialization_delay: int | None = None,
    ) -> None:
        self.total_tracker = total_tracker
        self.tracker = Tracker(
            filter_factory=OptimizedKalmanFilterFactory(),
            distance_function=distance_function,
            reid_distance_function=reid_distance_function
            if reid_distance_function is not None
            else self._embedding_distance,
            hit_counter_max=total_tracker * hit_counter_max,
            distance_threshold=distance_threshold,
            reid_hit_counter_max=total_tracker * reid_hit_counter_max,
            reid_distance_threshold=reid_distance_threshold,
            past_detections_length=past_detections_length,
            initialization_delay=initialization_delay,
        )

    def update(
        self, tracked: TrackedObject, projected_points: list[tuple[int, np.ndarray]]
    ):
        nf_detections = []
        for t_obj, (_, point) in zip(tracked, projected_points):
            nf_detections.append(
                Detection(
                    points=point[:2].reshape(1, -1),
                    scores=t_obj.last_detection.scores,
                    embedding=t_obj.last_detection.embedding,
                )
            )
        return self.tracker.update(nf_detections, self.total_tracker)

    def _embedding_distance(
        self,
        matched_not_init_trackers: TrackedObject,
        unmatched_trackers: TrackedObject,
    ):
        print(f"{time.time():.2f} Called REID")
        return mean_embedding_distance(matched_not_init_trackers, unmatched_trackers)
