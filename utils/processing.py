from functools import partial
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import openvino as ov
import supervision as sv
from cv2.typing import MatLike
from norfair import Detection, OptimizedKalmanFilterFactory, Tracker
from norfair.tracker import TrackedObject
from scipy.spatial.distance import cdist
from ultralytics import YOLO


class Detector:
    def __init__(self, model_path: Path, edge_margin: int = 20) -> None:
        self.model = YOLO(model_path)
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
    def __init__(self, model_path: Path) -> None:
        self.core = ov.Core()
        self.model = self.core.read_model(model_path)
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


def extract_embeddings(tracked_object: TrackedObject) -> list[np.ndarray | None]:
    embeddings = [tracked_object.last_detection.embedding]
    embeddings.extend(
        detection.embedding for detection in tracked_object.past_detections
    )
    return embeddings


def embedding_distance(
    matched_not_init_trackers: TrackedObject,
    unmatched_trackers: TrackedObject,
):
    snd_embeddings: list[np.ndarray] = extract_embeddings(unmatched_trackers)
    fst_embeddings: list[np.ndarray] = extract_embeddings(matched_not_init_trackers)

    if not snd_embeddings or not fst_embeddings:
        return 2.0

    distances = cdist(
        np.vstack(snd_embeddings), np.vstack(fst_embeddings), metric="cosine"
    )
    return np.min(distances)


def mean_embedding_distance(
    matched_not_init_trackers: TrackedObject, unmatched_trackers: TrackedObject
):
    snd_embeddings: list[np.ndarray] = extract_embeddings(unmatched_trackers)
    fst_embeddings: list[np.ndarray] = extract_embeddings(matched_not_init_trackers)

    if not snd_embeddings or not fst_embeddings:
        return 2.0

    snd_embedding = np.mean(np.vstack(snd_embeddings), axis=0, keepdims=True)
    fst_embedding = np.mean(np.vstack(fst_embeddings), axis=0, keepdims=True)

    return cdist(snd_embedding, fst_embedding, metric="cosine")[0, 0]


def infer_embeddings(reid_model: PersonRecognizer, tracked_object: TrackedObject):
    last_detection = tracked_object.last_detection
    if last_detection.embedding is None:
        last_detection.embedding = reid_model.infer(last_detection.data["cropped"])
    for detection in tracked_object.past_detections:
        if detection.embedding is None:
            detection.embedding = reid_model.infer(detection.data["cropped"])


class PersonTracker:
    def __init__(
        self,
        distance_function: str | Callable[[Detection, TrackedObject], float],
        reid_model: PersonRecognizer,
        reid_distance_function: Callable[[TrackedObject, TrackedObject], float]
        | None = None,
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
            reid_distance_function=reid_distance_function
            if reid_distance_function is not None
            else partial(self._embedding_distance, reid_model),
            reid_distance_threshold=reid_distance_threshold,
            reid_hit_counter_max=reid_hit_counter_max,
            past_detections_length=past_detections_length,
            initialization_delay=initialization_delay,
        )

    def update(
        self,
        detections: sv.Detections,
        cropped_objects: list[MatLike],
    ):
        nf_detections = [
            Detection(
                points=np.array([(xyxy[0] + xyxy[2]) / 2, xyxy[3]]),
                scores=np.array([conf]),
                data={"cropped": mat},
            )
            for xyxy, conf, mat in zip(
                detections.xyxy, detections.confidence, cropped_objects
            )
        ]
        return self.tracker.update(nf_detections)

    def _embedding_distance(
        self,
        reid_model: PersonRecognizer,
        matched_not_init_trackers: TrackedObject,
        unmatched_trackers: TrackedObject,
    ):
        infer_embeddings(reid_model, matched_not_init_trackers)
        infer_embeddings(reid_model, unmatched_trackers)
        return embedding_distance(matched_not_init_trackers, unmatched_trackers)
