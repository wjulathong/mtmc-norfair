from pathlib import Path

import cv2
import numpy as np
from cv2.typing import MatLike
from norfair.tracker import TrackedObject
from supervision import crop_image

from utils.drawing import (
    FloorPlanDrawer,
    GlobalFloorPlanDrawer,
    annotate,
    draw_text_on_frame,
    preview_frame,
)
from utils.global_association import GlobalMatcher
from utils.processing import Detector, PersonRecognizer, PersonTracker
from utils.transform import load_homography, prepare_floor_plan, project_points
from utils.video import FPS, FrameGetter

MAIN_WINDOW_NAME = "Camera"

YOLO_MODEL_PATH = Path("./models/yolo11m.pt")
REID_MODEL_NAME = "person-reidentification-retail-0288"
REID_MODEL_SIZE = "FP16"
REID_MODEL_PATH = Path(
    f"./models/intel/{REID_MODEL_NAME}/{REID_MODEL_SIZE}/{REID_MODEL_NAME}.xml"
)


def main() -> None:
    floor_plan = cv2.imread(str(Path("../mdx/building=Nvidia-Bldg-K-Map.png")))
    scaled_floor_plan, transform_matrix = prepare_floor_plan(floor_plan, 1.5)

    video_paths = [
        Path("../mdx/Building_K_Cam1.mp4"),
        Path("../mdx/Building_K_Cam2.mp4"),
        Path("../mdx/Building_K_Cam6.mp4"),
        Path("../mdx/Building_K_Cam7.mp4"),
    ]
    calibrated_paths = [
        Path("./calibrated/Cam1.json"),
        Path("./calibrated/Cam2.json"),
        Path("./calibrated/Cam6.json"),
        Path("./calibrated/Cam7.json"),
    ]
    for path in video_paths:
        assert path.exists()
    for path in calibrated_paths:
        assert path.exists()
    assert len(video_paths) == len(calibrated_paths), (
        "Please make sure all camera is calibrated"
        f", {len(video_paths)} cameras != {len(calibrated_paths)} calibrations"
    )

    # Model
    det = Detector(YOLO_MODEL_PATH)
    rec = PersonRecognizer(REID_MODEL_PATH)

    caps = [FrameGetter(video_path, 5) for video_path in video_paths]
    trks = [
        PersonTracker("euclidean", rec, reid_distance_threshold=0.45)
        for _ in video_paths
    ]
    homo_mats = [load_homography(path) for path in calibrated_paths]
    projections: dict[int, list[tuple[TrackedObject, np.ndarray]]] = {}

    global_matcher = GlobalMatcher(rec, reid_threshold=0.7)

    # Visualizer
    floor_plan_drawer = FloorPlanDrawer(
        scaled_floor_plan,
        history_length=50,
        transform_matrix=transform_matrix,
    )
    global_floor_plan_drawer = GlobalFloorPlanDrawer(
        scaled_floor_plan,
        history_length=50,
        transform_matrix=transform_matrix,
    )

    manual = False
    paused = False
    fps = FPS()
    last_frames: dict[int, MatLike | None] = {}
    while True:
        if not paused or manual:
            frames = [(cid, cap.read()) for cid, cap in enumerate(caps)]
            if any(frame is None for _, frame in frames):
                break

            for cid, frame in frames:
                detections = det.detect(frame)
                obj_mats: list[MatLike] = [
                    crop_image(frame, xyxy) for xyxy in detections.xyxy
                ]
                tracked_objects = trks[cid].update(detections, obj_mats)
                projections[cid] = project_points(tracked_objects, homo_mats[cid])
                last_frame = preview_frame(
                    annotate(frame, detections, tracked_objects), 0.3
                )
                last_frames[cid] = last_frame
                if manual:
                    draw_text_on_frame(last_frame, "Manual")
                cv2.imshow(f"{MAIN_WINDOW_NAME}_{cid}", last_frame)

            global_objects = global_matcher.match(projections)

            locals_frame = preview_frame(floor_plan_drawer.draw(projections), 0.3)
            global_frame = preview_frame(
                global_floor_plan_drawer.draw(global_objects), 0.3
            )
            if manual:
                draw_text_on_frame(locals_frame, "Manual")
                draw_text_on_frame(global_frame, "Manual")
            cv2.imshow("Locals", locals_frame)
            cv2.imshow("Global", global_frame)

            manual = False

        fps.update()
        if not paused:
            print(f"FPS: {fps.fps:.2f}")

        key_press = cv2.waitKey(1) & 0xFF
        if key_press == ord("p"):
            paused = not paused
            if paused:
                for cid, last_frame in last_frames.items():
                    draw_text_on_frame(last_frame, "Paused")
                    cv2.imshow(f"{MAIN_WINDOW_NAME}_{cid}", last_frame)
                draw_text_on_frame(locals_frame, "Paused")
                draw_text_on_frame(global_frame, "Paused")
                cv2.imshow("Locals", locals_frame)
                cv2.imshow("Global", global_frame)
        elif key_press == ord("e"):
            manual = True
        elif key_press == ord("d"):
            print("\nprojections")
            print(projections)
            print("\nglobal_objects")
            print(global_matcher.global_objects)
            print("\nglobal -> locals")
            for global_id, global_obj in global_matcher.global_objects.items():
                local_objs = [
                    f"{cid}_{lobj.tracked_object.id}"
                    for cid, lobj in global_obj.cameras.items()
                ]
                print(f"{global_id} -> [{', '.join(local_objs)}]")
            print()
        elif key_press == ord("l"):
            for cap in caps:
                cap.seek_forward(10)
        elif key_press == ord("h"):
            for cap in caps:
                cap.seek_backward(10)
        elif key_press in (27, ord("q")):
            break
        elif (
            cv2.getWindowProperty(f"{MAIN_WINDOW_NAME}_{cid}", cv2.WND_PROP_VISIBLE) < 1
        ):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
