from dataclasses import dataclass, field

import numpy as np
from norfair.tracker import TrackedObject
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from .processing import PersonRecognizer, extract_embeddings, infer_embeddings


@dataclass(slots=True)
class LocalObject:
    tracked_object: TrackedObject
    projected_position: np.ndarray
    frame_time: int
    global_id: int | None = None

    def is_active(self):
        return (
            not self.tracked_object.is_initializing
            and self.tracked_object.hit_counter_is_positive
        )

    def is_dead(self, current_time: int):
        return current_time > self.frame_time and not self.is_active()

    def get_embeddings(self, reid_model: PersonRecognizer):
        embeddings = extract_embeddings(self.tracked_object)
        if any(emb is None for emb in embeddings):
            infer_embeddings(reid_model, self.tracked_object)
            embeddings = extract_embeddings(self.tracked_object)
        return embeddings


@dataclass(slots=True)
class GlobalObject:
    id: int
    last_frame_time: int
    cameras: dict[int, LocalObject] = field(default_factory=dict)
    position: np.ndarray | None = None
    staled: bool = False

    def update(self, current_time: int):
        if all(local_obj.is_dead(current_time) for local_obj in self.cameras.values()):
            if current_time - self.last_frame_time >= 1800:  # TODO: To be configurable
                self.staled = True
            return
        else:
            self.last_frame_time = current_time
            self.update_position()

    def update_position(self):
        positions = np.array([obj.projected_position for obj in self.cameras.values()])

        if positions.shape[0] == 1:
            self.position = positions[0]
            return

        hit_counters = np.array(
            [obj.tracked_object.hit_counter for obj in self.cameras.values()]
        )
        weights = np.exp(0.5 * hit_counters)
        norm_weights = weights / np.sum(weights)
        position = np.sum(positions * norm_weights[:, None], axis=0)
        self.position = position

    def get_embeddings(self, reid_model: PersonRecognizer):
        embeddings = []
        for local_obj in self.cameras.values():
            embeddings.extend(local_obj.get_embeddings(reid_model))
        return embeddings


class GlobalMatcher:
    INVALID_COST: float = 1000.0

    def __init__(
        self,
        reid_model: PersonRecognizer,
        reid_threshold: float | tuple[float, float] = 0.7,
        pos_threshold: float = 200.0,
        reid_weight: float = 0.5,
        pos_weight: float = 0.5,
    ):
        self.reid_model = reid_model
        self.reid_threshold = reid_threshold
        self.pos_threshold = pos_threshold
        self.reid_weight = reid_weight
        self.pos_weight = pos_weight
        self.global_objects: dict[int, GlobalObject] = {}
        self.global_id_counter = 1

    def _get_new_global_id(self):
        global_id = self.global_id_counter
        self.global_id_counter += 1
        return global_id

    def get_active_objects(self):
        return [
            global_obj
            for global_obj in self.global_objects.values()
            if any(local_obj.is_active() for local_obj in global_obj.cameras.values())
        ]

    def match(
        self,
        camera_id: int,
        current_time: int,
        tracked_objects: list[TrackedObject],
        projected_positions: list[np.ndarray],
    ):
        local_objects: list[LocalObject] = []
        for obj, pos in zip(tracked_objects, projected_positions):
            local_obj = LocalObject(
                tracked_object=obj,
                projected_position=pos,
                frame_time=current_time,
            )
            for global_obj in self.global_objects.values():
                if (
                    camera_id in global_obj.cameras
                    and global_obj.cameras[camera_id].tracked_object.id == obj.id
                ):
                    local_obj.global_id = global_obj.id
                    global_obj.cameras[camera_id] = local_obj
                    break
            local_objects.append(local_obj)

        self._match_camera(camera_id, current_time, local_objects)

        for global_obj in self.global_objects.values():
            global_obj.update(current_time)

        staled_objs = [
            global_obj.id
            for global_obj in self.global_objects.values()
            if global_obj.staled
        ]
        for global_id in staled_objs:
            del self.global_objects[global_id]

    def _match_camera(
        self, camera_id: int, current_time: int, local_objects: list[LocalObject]
    ):
        if not local_objects:
            return

        # Get all unmatched local objects
        unmatched_locals = [
            local_obj for local_obj in local_objects if local_obj.global_id is None
        ]

        if not unmatched_locals:
            # TODO: Refinded matched objects
            return

        # If no existing global objects, create new ones for all unmatched objects
        if not self.global_objects:
            for local_obj in unmatched_locals:
                self._create_new_global_object(local_obj, camera_id, current_time)
            return

        global_objects = list(self.global_objects.values())

        # Check for two stages threshold
        # Stage 1: Confidence ReID
        if isinstance(self.reid_threshold, tuple):
            reid_distances = self._compute_reid_distances(
                unmatched_locals, global_objects
            )
            cost_matrix = reid_distances / self.reid_threshold[0]
            cost_matrix[reid_distances > self.reid_threshold[0]] = self.INVALID_COST
        else:
            pos_distances, reid_distances = self._calculate_distances(
                unmatched_locals, global_objects
            )
            cost_matrix = self._compute_cost_matrix(
                pos_distances,
                reid_distances,
                self.pos_threshold,
                self.reid_threshold,
                self.pos_weight,
                self.reid_weight,
            )

        matched_local_indices = self._process_matches(
            camera_id, unmatched_locals, global_objects, cost_matrix
        )

        # Stage 2: Lower confidence ReID with position
        unmatched_locals = [
            local_obj
            for i, local_obj in enumerate(unmatched_locals)
            if i not in matched_local_indices
        ]

        if isinstance(self.reid_threshold, tuple) and unmatched_locals:
            pos_distances, reid_distances = self._calculate_distances(
                unmatched_locals,
                global_objects,
            )
            cost_matrix = self._compute_cost_matrix(
                pos_distances,
                reid_distances,
                self.pos_threshold,
                self.reid_threshold[1],
                self.pos_weight,
                self.reid_weight,
            )
            matched_local_indices = self._process_matches(
                camera_id, unmatched_locals, global_objects, cost_matrix
            )
            unmatched_locals = [
                local_obj
                for i, local_obj in enumerate(unmatched_locals)
                if i not in matched_local_indices
            ]

        # Create new global objects for remaining unmatched local objects
        for local_obj in unmatched_locals:
            if local_obj.global_id is None:
                self._create_new_global_object(local_obj, camera_id, current_time)

    def _calculate_distances(
        self, unmatched_locals: list[LocalObject], global_objects: list[GlobalObject]
    ):
        # Calculate position distances
        local_positions = [obj.projected_position for obj in unmatched_locals]
        global_positions = [obj.position for obj in global_objects]
        pos_distances = cdist(
            np.array(local_positions), np.array(global_positions), metric="euclidean"
        )
        reid_distances = self._compute_reid_distances(unmatched_locals, global_objects)
        return pos_distances, reid_distances

    def _compute_reid_distances(
        self, unmatched_locals: list[LocalObject], global_objects: list[GlobalObject]
    ) -> np.ndarray:
        # Calculate ReID distances using all available embeddings
        reid_distances = np.zeros((len(unmatched_locals), len(global_objects)))
        for i, local_obj in enumerate(unmatched_locals):
            for j, global_obj in enumerate(global_objects):
                reid_distances[i, j] = self._compute_reid_distance(
                    local_obj.get_embeddings(self.reid_model),
                    global_obj.get_embeddings(self.reid_model),
                )
        return reid_distances

    def _compute_cost_matrix(
        self,
        pos_distances: np.ndarray,
        reid_distances: np.ndarray,
        pos_threshold: float,
        reid_threshold: float,
        pos_weight: float,
        reid_weight: float,
    ):
        # Combine distances with weights
        cost_matrix = pos_weight * (pos_distances / pos_threshold) + reid_weight * (
            reid_distances / reid_threshold
        )
        # Mark invalid matches with a high cost
        invalid_matches = (pos_distances > pos_threshold) | (
            reid_distances > reid_threshold
        )
        cost_matrix[invalid_matches] = self.INVALID_COST
        return cost_matrix

    def _process_matches(
        self,
        camera_id: int,
        unmatched_locals: list[LocalObject],
        global_objects: list[GlobalObject],
        cost_matrix: np.ndarray,
    ):
        # Apply Hungarian algorithm
        local_indices, assignment_indices = linear_sum_assignment(cost_matrix)

        matched_local_indices: list[int] = []
        # Process matches
        for local_idx, assign_idx in zip(local_indices, assignment_indices):
            local_idx: int
            assign_idx: int
            # Skip invalid matches (those with high cost)
            if cost_matrix[local_idx, assign_idx] >= self.INVALID_COST:
                continue

            local_obj = unmatched_locals[local_idx]
            global_obj = global_objects[assign_idx]

            # Update assignments
            local_obj.global_id = global_obj.id
            global_obj.cameras[camera_id] = local_obj
            matched_local_indices.append(local_idx)
        return matched_local_indices

    def _compute_reid_distance(
        self, local_embeddings: list[np.ndarray], global_embeddings: list[np.ndarray]
    ):
        # Compute single ReID distance between two objects
        distances = cdist(
            np.vstack(local_embeddings), np.vstack(global_embeddings), metric="cosine"
        )
        return np.min(distances)

    def _create_new_global_object(
        self, local_obj: LocalObject, camera_id: int, current_time: int
    ):
        global_id = self._get_new_global_id()
        global_obj = GlobalObject(id=global_id, last_frame_time=current_time)
        global_obj.cameras[camera_id] = local_obj
        self.global_objects[global_id] = global_obj
