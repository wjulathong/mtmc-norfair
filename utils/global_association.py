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
    global_id: int | None = None

    def get_embeddings(self, reid_model: PersonRecognizer):
        embeddings = extract_embeddings(self.tracked_object)
        if any(emb is None for emb in embeddings):
            infer_embeddings(reid_model, self.tracked_object)
            embeddings = extract_embeddings(self.tracked_object)
        return embeddings


@dataclass(slots=True)
class GlobalObject:
    id: int
    cameras: dict[int, LocalObject] = field(default_factory=dict)
    position: np.ndarray | None = None

    def update_position(self):
        if not self.cameras:
            return

        positions = [obj.projected_position for obj in self.cameras.values()]
        self.position = np.mean(positions, axis=0)

    def get_embeddings(self, reid_model: PersonRecognizer):
        embeddings = []
        for local_obj in self.cameras.values():
            embeddings.extend(local_obj.get_embeddings(reid_model))
        return embeddings


class GlobalMatcher:
    def __init__(
        self,
        reid_model: PersonRecognizer,
        reid_threshold: float = 0.7,
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

    def match(self, projections: dict[int, list[tuple[TrackedObject, np.ndarray]]]):
        local_objects_by_camera: dict[int, dict[int, LocalObject]] = {}

        for cid, objs in projections.items():
            local_objects = {}
            for obj, projected_position in objs:
                local_obj = LocalObject(
                    tracked_object=obj,
                    projected_position=projected_position,
                )

                for global_obj in self.global_objects.values():
                    if (
                        cid in global_obj.cameras
                        and global_obj.cameras[cid].tracked_object.id == obj.id
                    ):
                        local_obj.global_id = global_obj.id
                        global_obj.cameras[cid] = local_obj
                        break

                local_objects[obj.id] = local_obj
            local_objects_by_camera[cid] = local_objects

            self._match_camera(cid, local_objects)

        # Final position update for all global objects after all cameras are processed
        for global_obj in self.global_objects.values():
            global_obj.update_position()

        return list(self.global_objects.values())

    def _match_camera(self, camera_id: int, local_objects: dict[int, LocalObject]):
        if not local_objects:
            return

        # Get all unmatched local objects
        unmatched_locals = {
            local_id: local_obj
            for local_id, local_obj in local_objects.items()
            if local_obj.global_id is None
        }

        if not unmatched_locals:
            return

        # If no existing global objects, create new ones for all unmatched objects
        if not self.global_objects:
            for local_obj in unmatched_locals.values():
                self._create_new_global_object(local_obj, camera_id)
            return

        # Prepare arrays for cost matrix
        local_list = list(unmatched_locals.values())
        global_list = list(self.global_objects.values())

        n_local = len(local_list)
        n_global = len(global_list)

        # Calculate position distances
        local_positions = [obj.projected_position for obj in local_list]
        global_positions = [obj.position for obj in global_list]
        pos_distances = cdist(
            np.array(local_positions), np.array(global_positions), metric="euclidean"
        )

        # Calculate ReID distances using all available embeddings
        reid_distances = np.zeros((n_local, n_global))

        for i, local_obj in enumerate(local_list):
            for j, global_obj in enumerate(global_list):
                reid_distances[i, j] = self._compute_reid_distance(
                    local_obj.get_embeddings(self.reid_model),
                    global_obj.get_embeddings(self.reid_model),
                )


        # Combine distances with weights
        cost_matrix = self.pos_weight * (
            pos_distances / self.pos_threshold
        ) + self.reid_weight * (reid_distances / self.reid_threshold)

        # Mark invalid matches with a high cost
        invalid_matches = (pos_distances > self.pos_threshold) | (
            reid_distances > self.reid_threshold
        )
        cost_matrix[invalid_matches] = 1000.0

        # Apply Hungarian algorithm
        local_indices, assignment_indices = linear_sum_assignment(cost_matrix)

        # Process matches
        for local_idx, assign_idx in zip(local_indices, assignment_indices):
            # Skip invalid matches (those with high cost)
            if cost_matrix[local_idx, assign_idx] >= 1000.0:
                continue

            local_obj = local_list[local_idx]
            global_obj = global_list[assign_idx]

            # Update assignments
            local_obj.global_id = global_obj.id
            global_obj.cameras[camera_id] = local_obj
            global_obj.update_position()

        # Create new global objects for remaining unmatched local objects
        for local_obj in local_list:
            if local_obj.global_id is None:
                self._create_new_global_object(local_obj, camera_id)

    def _compute_reid_distance(
        self, local_embeddings: list[np.ndarray], global_embeddings: list[np.ndarray]
    ):
        local_embeddings_array = np.array(local_embeddings).reshape(-1, 256)
        global_embeddings_array = np.array(global_embeddings).reshape(-1, 256)
        distances = cdist(
            local_embeddings_array, global_embeddings_array, metric="cosine"
        )
        return np.min(distances)

    def _create_new_global_object(self, local_obj: LocalObject, camera_id: int):
        global_id = self._get_new_global_id()
        global_obj = GlobalObject(id=global_id)
        global_obj.cameras[camera_id] = local_obj
        local_obj.global_id = global_id
        global_obj.update_position()
        self.global_objects[global_id] = global_obj
