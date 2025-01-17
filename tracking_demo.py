from pathlib import Path

import cv2
import numpy as np
from cv2.typing import MatLike
from norfair.tracker import TrackedObject
from supervision import crop_image

from utils.drawing import FloorPlanDrawer, annotate, draw_floor_plan, preview_frame
from utils.processing import Detector, GlobalTracker, PersonRecognizer, PersonTracker
from utils.transform import load_homography, prepare_floor_plan, project_points
from utils.video import FPS, FrameGetter

MAIN_WINDOW_NAME = "Camera"

YOLO_MODEL_PATH = Path("./models/yolo11s.pt")
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
        Path("../mdx/Building_K_Cam6.mp4"),
        Path("../mdx/Building_K_Cam7.mp4"),
    ]
    calibrated_paths = [
        Path("./calibrated/Cam1.json"),
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
    trks = [PersonTracker("euclidean", rec) for _ in video_paths]
    global_tracker = GlobalTracker(len(caps), "euclidean", rec, distance_threshold=50)
    homo_mats = [load_homography(path) for path in calibrated_paths]
    projections: dict[int, tuple[TrackedObject, list[int, np.ndarray]]] = {}
    global_tracked: list[TrackedObject] = []

    # Visualizer
    floor_plan_drawer = FloorPlanDrawer(
        scaled_floor_plan, history_length=50, transform_matrix=transform_matrix
    )

    manual = False
    paused = False
    fps = FPS()
    last_frames: dict[int, MatLike | None] = {}
    while True:
        if not paused or manual:
            manual = False
            frames = [(cid, cap.read()) for cid, cap in enumerate(caps)]
            if any(frame is None for _, frame in frames):
                break

            for cid, frame in frames:
                detections = det.detect(frame)
                obj_mats: list[MatLike] = [
                    crop_image(frame, xyxy) for xyxy in detections.xyxy
                ]
                tracked = trks[cid].update(detections, obj_mats)
                projections[cid] = (tracked, project_points(tracked, homo_mats[cid]))
                last_frames[cid] = preview_frame(
                    annotate(frame, detections, tracked), 0.4
                )

        fps.update()
        print(f"FPS: {fps.fps:.2f}")

        for cid, frame in last_frames.items():
            if frame is None:
                break
            all_projected = preview_frame(
                draw_floor_plan(scaled_floor_plan, projections, transform_matrix),
                0.4,
            )
            if paused:
                cv2.putText(
                    frame,
                    "Paused",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    all_projected,
                    "Paused",
                    (10, all_projected.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
            cv2.imshow(f"{MAIN_WINDOW_NAME}_{cid}", frame)
            cv2.imshow("all_projected", all_projected)

        if not paused or manual:
            for cid, (t_obj, projected_points) in projections.items():
                global_tracked = global_tracker.update(t_obj, projected_points)

        if global_tracked:
            fp_frame = preview_frame(floor_plan_drawer.draw(global_tracked), 0.4)
            if paused:
                cv2.putText(
                    fp_frame,
                    "Paused",
                    (10, fp_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
            cv2.imshow("global_map", fp_frame)

        key_press = cv2.waitKey(1) & 0xFF
        if key_press == ord("p"):
            paused = not paused
        elif key_press == ord("e"):
            manual = True
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
