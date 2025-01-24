from functools import partial
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import supervision as sv
import torch
from torch.nn.functional import normalize
from cv2.typing import MatLike
from norfair import Detection, OptimizedKalmanFilterFactory, Tracker
from norfair.tracker import TrackedObject
from scipy.spatial.distance import cdist
from ultralytics import YOLO

from fastreid.config import get_cfg
from fastreid.engine.defaults import DefaultPredictor


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
    def __init__(self, config_file: Path, opts: list | None = None) -> None:
        opts = [] if opts is None else opts
        cfg = get_cfg()
        cfg.merge_from_file(str(config_file))
        cfg.merge_from_list(opts)
        self.cfg = cfg
        self.predictor = DefaultPredictor(cfg)

    def infer(self, frame: MatLike) -> np.ndarray:
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (np.ndarray): normalized feature of the model.
        """
        # the model expects RGB inputs
        frame = frame[:, :, ::-1]
        # Apply pre-processing to image.
        image = cv2.resize(
            frame,
            tuple(self.cfg.INPUT.SIZE_TEST[::-1]),
            interpolation=cv2.INTER_CUBIC,
        )
        # Make shape with a new batch dimension which is adapted for
        # network input
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))[None]
        predictions = self.predictor(image)
        return normalize(predictions).data.numpy()


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

    emb_size = snd_embeddings[0].shape[1]
    snd_embeddings = np.array(snd_embeddings).reshape(-1, emb_size)
    fst_embeddings = np.array(fst_embeddings).reshape(-1, emb_size)

    distances = cdist(snd_embeddings, fst_embeddings, metric="cosine")
    return np.min(distances)


def mean_embedding_distance(
    matched_not_init_trackers: TrackedObject, unmatched_trackers: TrackedObject
):
    snd_embeddings: list[np.ndarray] = extract_embeddings(unmatched_trackers)
    fst_embeddings: list[np.ndarray] = extract_embeddings(matched_not_init_trackers)

    if not snd_embeddings or not fst_embeddings:
        return 2.0

    emb_size = snd_embeddings[0].shape[1]
    snd_embeddings = np.array(snd_embeddings).reshape(-1, emb_size)
    fst_embeddings = np.array(fst_embeddings).reshape(-1, emb_size)

    snd_embedding = np.mean(snd_embeddings, axis=0).reshape(1, -1)
    fst_embedding = np.mean(fst_embeddings, axis=0).reshape(1, -1)

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
