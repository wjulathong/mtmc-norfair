from pathlib import Path

import cv2
import norfair
import numpy as np
import supervision as sv
from cv2.typing import MatLike
from norfair.tracker import TrackedObject

from processors import Detector, PersonRecognizer, PersonTracker
from utils import (
    FrameGetter,
    draw_floor_plan,
    load_homography,
    prepare_floor_plan,
    preview_frame,
    project_points,
)

MAIN_WINDOW_NAME = "Camera"

BBOX_ANN = sv.BoxAnnotator(thickness=3)
LABEL_ANN = sv.LabelAnnotator()


def annotate(frame: MatLike, detections: sv.Detections, tracked: list[TrackedObject]):
    labels = [
        f"{cname} {conf:.2f}"
        for cname, conf in zip(detections["class_name"], detections.confidence)
    ]
    annot_frame = BBOX_ANN.annotate(scene=frame.copy(), detections=detections)
    annot_frame = LABEL_ANN.annotate(
        scene=annot_frame, detections=detections, labels=labels
    )
    norfair.draw_points(annot_frame, tracked, radius=8, text_size=1)
    return annot_frame


def main() -> None:
    floor_plan = cv2.imread(str(Path("../mdx/building=Nvidia-Bldg-K-Map.png")))
    scaled_floor_plan, transform_matrix = prepare_floor_plan(floor_plan, 1.5)

    video_paths = [
        Path("../mdx/Building_K_Cam1.mp4"),
        Path("../mdx/Building_K_Cam7.mp4"),
    ]
    calibrated_paths = [
        Path("./calibrated/Cam1.json"),
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

    caps = [FrameGetter(video_path, 5) for video_path in video_paths]
    trks = [PersonTracker() for _ in video_paths]
    homo_mats = [load_homography(path) for path in calibrated_paths]
    projections: dict[int, list[int, np.ndarray]] = {}

    # Model
    det = Detector()
    rec = PersonRecognizer()

    manual = False
    paused = False
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
                    sv.crop_image(image=frame, xyxy=xyxy) for xyxy in detections.xyxy
                ]
                reids = [rec.infer(mat) for mat in obj_mats]
                tracked = trks[cid].update(detections, reids)
                projections[cid] = project_points(tracked, homo_mats[cid])
                last_frames[cid] = preview_frame(
                    annotate(frame, detections, tracked), 0.4
                )

        for cid, frame in last_frames.items():
            if frame is None:
                break
            fp_frame = preview_frame(
                draw_floor_plan(scaled_floor_plan, projections, transform_matrix),
                0.7,
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
                    fp_frame,
                    "Paused",
                    (10, fp_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
            cv2.imshow(f"{MAIN_WINDOW_NAME}_{cid}", frame)
            cv2.imshow("floor_plan", fp_frame)

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
