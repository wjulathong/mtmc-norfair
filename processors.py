from pathlib import Path

import cv2
import numpy as np
import openvino as ov
import supervision as sv
from cv2.typing import MatLike
from norfair import Detection, OptimizedKalmanFilterFactory, Tracker
from norfair.tracker import TrackedObject
from scipy.spatial.distance import cosine
from ultralytics import YOLO


class Detector:
    def __init__(self) -> None:
        self.model_path = Path("./models/yolo11s.pt")
        self.model = YOLO(self.model_path)

    def detect(self, frame: MatLike) -> sv.Detections:
        results = self.model.predict(source=frame, conf=0.6, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[detections["class_name"] == "person"]
        return detections


class PersonRecognizer:
    def __init__(self) -> None:
        self.core = ov.Core()
        self.model = self.core.read_model(
            Path(
                "./models/intel/person-reidentification-retail-0288/FP16/person-reidentification-retail-0288.xml"
            )
        )
        self.model.reshape([1, 3, 256, 128])
        self.compiled_model = self.core.compile_model(self.model)

    def infer(self, mat: MatLike):
        tensor = self._create_input_tensor(mat)
        return self.compiled_model(tensor)[0]

    def _create_input_tensor(self, mat: MatLike):
        input_tensor: ov.runtime.ConstOutput = self.compiled_model.input()
        tensor_shape: ov.Shape = input_tensor.shape
        layout: ov.Layout = ov.Layout("NCHW")
        width: int = tensor_shape[layout.get_index_by_name("width")]
        height: int = tensor_shape[layout.get_index_by_name("height")]
        resized_mat = cv2.resize(mat, (width, height))
        return ov.Tensor(
            np.expand_dims(resized_mat.transpose(2, 0, 1).astype(np.float32), axis=0)
        )


class PersonTracker:
    def __init__(self) -> None:
        self.count = 0
        self.tracker = Tracker(
            # initialization_delay=1,
            distance_function="euclidean",
            hit_counter_max=10,
            filter_factory=OptimizedKalmanFilterFactory(),
            distance_threshold=50,
            past_detections_length=10,
            reid_distance_function=self._embedding_distance,
            reid_distance_threshold=0.7,
            reid_hit_counter_max=300,
        )

    def update(self, detections: sv.Detections, reids: list[np.ndarray]):
        nf_detections = []
        for xyxy, conf, emb in zip(detections.xyxy, detections.confidence, reids):
            points = np.array([(xyxy[0] + xyxy[2]) / 2, xyxy[3]])
            scores = np.array([conf])
            nf_detections.append(Detection(points=points, scores=scores, embedding=emb))
        return self.tracker.update(nf_detections)

    def _embedding_distance(
        self,
        matched_not_init_trackers: TrackedObject,
        unmatched_trackers: TrackedObject,
    ):
        # print(f"{self.count} Ran REID Distance Calculatation")
        # print(matched_not_init_trackers.id, unmatched_trackers.id)
        self.count += 1
        # Collect all valid embeddings from unmatched tracker
        snd_embeddings = []

        # Add last detection if it has embedding
        if unmatched_trackers.last_detection.embedding is not None:
            snd_embeddings.append(
                np.array(unmatched_trackers.last_detection.embedding).reshape(256)
            )

        # Add all past detections with valid embeddings
        for detection in unmatched_trackers.past_detections:
            if detection.embedding is not None:
                snd_embeddings.append(np.array(detection.embedding).reshape(256))

        # If no valid embeddings found at all, return maximum distance
        if not snd_embeddings:
            return 2.0

        # Collect all valid embeddings from matched tracker
        fst_embeddings = []
        for detection_fst in matched_not_init_trackers.past_detections:
            if detection_fst.embedding is not None:
                fst_embeddings.append(np.array(detection_fst.embedding).reshape(256))

        # If no valid embeddings in matched tracker, return maximum distance
        if not fst_embeddings:
            return 2.0

        # Calculate distances between all pairs of embeddings
        min_distance = 2.0
        for snd_emb in snd_embeddings:
            for fst_emb in fst_embeddings:
                distance = cosine(snd_emb, fst_emb)
                min_distance = min(min_distance, distance)

        # print(min_distance)
        return min_distance

    def _embedding_distance_old(
        self,
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
                return 1.0

        snd_embedding = np.array(snd_embedding).reshape(256)

        for detection_fst in matched_not_init_trackers.past_detections:
            if detection_fst.embedding is None:
                continue

            fst_embedding = np.array(detection_fst.embedding).reshape(256)

            dist = cosine(snd_embedding, fst_embedding)
            return dist
        return 1.0
