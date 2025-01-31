import json
from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np
from cv2.typing import MatLike

FLOOR_PLAN_IMAGE_PATH = Path("../mdx/building=Nvidia-Bldg-K-Map.png")

CAMERA_NAME = "Cam6"
CAMERA_IMAGE_PATH = Path(f"../mdx/{CAMERA_NAME}_Image.png")
HOMOGRAPHY_PATH = Path(f"./calibrated/{CAMERA_NAME}.json")

FIXED_PREVIEW_WIDTH = 1740
FPS = 30


def resize_fixed_width(img: MatLike, fixed_width: int):
    aspect_ratio = img.shape[0] / img.shape[1]
    new_height = int(fixed_width * aspect_ratio)
    return cv2.resize(img, (fixed_width, new_height), interpolation=cv2.INTER_AREA)


def get_preview_image(
    c_img: MatLike, fp_img: MatLike, fixed_width: int | None = None
) -> MatLike:
    desired_height = max(c_img.shape[0], fp_img.shape[0])
    left_img = cv2.resize(
        c_img, (int(c_img.shape[1] * desired_height / c_img.shape[0]), desired_height)
    )
    right_img = cv2.resize(
        fp_img,
        (int(fp_img.shape[1] * desired_height / fp_img.shape[0]), desired_height),
    )
    sp_img = np.full((desired_height, 20, 3), (0, 0, 0), dtype=np.uint8)

    preview_img = cv2.hconcat([left_img, sp_img, right_img])
    cv2.putText(
        preview_img,
        "C: Calibrate, S: Save",
        (10, preview_img.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        4,
        (255, 128, 0),
        7,
    )
    return (
        resize_fixed_width(preview_img, fixed_width)
        if fixed_width is not None
        else preview_img
    )


class Data(TypedDict):
    winname: str
    ori_image: MatLike
    image: MatLike
    points: list[tuple[int, int]]
    resized_shape: tuple[int, int]


class AppData(TypedDict):
    camera: Data
    floor_plan: Data
    preview: Data


def convert_coords(
    x: int, y: int, original_shape: tuple[int, int], resized_shape: tuple[int, int]
) -> tuple[int, int]:
    w_scale = original_shape[1] / resized_shape[1]
    h_scale = original_shape[0] / resized_shape[0]
    return int(x * w_scale), int(y * h_scale)


def draw_markers(
    image: MatLike, color: tuple[int, int, int], points: list[tuple[int, int]]
) -> MatLike:
    cv2.polylines(
        image,
        [np.array(points, dtype=np.int32)],
        isClosed=True,
        color=color,
        thickness=2,
    )
    for i, point in enumerate(points, start=1):
        cv2.circle(image, point, radius=5, color=color, thickness=-1)
        cv2.putText(
            image,
            str(i),
            (point[0] + 10, point[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )
    return image


def mouse_event(event: int, x: int, y: int, _, param: tuple[str, AppData]):
    if event not in (1, 2):
        return
    winname = param[0]
    data: Data = param[1][winname]
    points = data["points"]
    if event == 1:
        points.append(convert_coords(x, y, data["image"].shape, data["resized_shape"]))
    else:
        if points:
            points.pop()
    color = (0, 0, 255) if winname == "camera" else (255, 0, 0)
    data["image"] = draw_markers(data["ori_image"].copy(), color, points)
    param[1]["preview"]["image"] = get_preview_image(
        param[1]["camera"]["image"],
        param[1]["floor_plan"]["image"],
        FIXED_PREVIEW_WIDTH,
    )


def calculate_reprojection_error(
    homography_matrix: np.ndarray,
    camera_points: np.ndarray,
    floor_plan_points: np.ndarray,
) -> tuple[float, list[float]]:
    camera_homogeneous = np.hstack([camera_points, np.ones([len(camera_points), 1])])
    print(camera_homogeneous)
    transformed_points = camera_homogeneous @ homography_matrix.T
    transformed_points = transformed_points[:, :2] / transformed_points[:, 2:]
    print(transformed_points)
    errors = np.sqrt(np.sum((transformed_points - floor_plan_points) ** 2, axis=1))
    print(errors)
    return float(np.mean(errors)), errors.tolist()


def save_data(
    homography_matrix: np.ndarray,
    camera_points: np.ndarray,
    floor_plan_points: np.ndarray,
    mean_reprojection_error: float,
    reprojection_errors: list[float],
) -> None:
    data = {
        "homography_matrix": homography_matrix.tolist(),
        "camera_points": camera_points.tolist(),
        "floor_plan_points": floor_plan_points.tolist(),
        "mean_reprojection_error": mean_reprojection_error,
        "reprojection_errors": reprojection_errors,
    }
    with HOMOGRAPHY_PATH.open("w") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    winnames = ["camera", "floor_plan", "preview"]
    c_img = cv2.imread(str(CAMERA_IMAGE_PATH))
    fp_img = cv2.imread(str(FLOOR_PLAN_IMAGE_PATH))
    preview_img = get_preview_image(c_img, fp_img, FIXED_PREVIEW_WIDTH)

    calibrated = HOMOGRAPHY_PATH.exists()

    if calibrated:
        with HOMOGRAPHY_PATH.open() as f:
            read_data = json.load(f)
        camera_points = [tuple(map(int, point)) for point in read_data["camera_points"]]
        floor_plan_points = [
            tuple(map(int, point)) for point in read_data["floor_plan_points"]
        ]
        drawn_c_img = draw_markers(c_img.copy(), (0, 0, 255), camera_points)
        drawn_fp_img = draw_markers(fp_img.copy(), (255, 0, 0), floor_plan_points)

    images = {
        "camera": drawn_c_img if calibrated else c_img,
        "floor_plan": drawn_fp_img if calibrated else fp_img,
        "preview": get_preview_image(drawn_c_img, drawn_fp_img, FIXED_PREVIEW_WIDTH)
        if calibrated
        else preview_img,
    }
    points = {
        "camera": camera_points if calibrated else [],
        "floor_plan": floor_plan_points if calibrated else [],
        "preview": [],
    }

    app_data: AppData = {
        winname: Data(
            ori_image=img,
            image=images[winname],
            points=points[winname],
            resized_shape=resize_fixed_width(img, FIXED_PREVIEW_WIDTH).shape,
        )
        for winname, img in zip(winnames, [c_img, fp_img, preview_img])
    }

    cv2.namedWindow("camera")
    cv2.namedWindow("floor_plan")
    cv2.setMouseCallback("camera", mouse_event, param=["camera", app_data])
    cv2.setMouseCallback("floor_plan", mouse_event, param=["floor_plan", app_data])

    while True:
        timer = cv2.getTickCount()

        for winname in winnames:
            data: Data = app_data[winname]
            cv2.imshow(winname, resize_fixed_width(data["image"], FIXED_PREVIEW_WIDTH))

        delay = (1000 // FPS) - int(
            (cv2.getTickCount() - timer) / cv2.getTickFrequency() * 1000
        )
        key_press = cv2.waitKey(max(1, delay)) & 0xFF
        if key_press in (ord("c"), ord("s")):
            c_camera_points = len(app_data["camera"]["points"])
            c_floor_plan_points = len(app_data["floor_plan"]["points"])
            if c_camera_points != c_floor_plan_points:
                print(
                    f"Mismatch points: {c_camera_points} pts in Camera View"
                    f", but {c_floor_plan_points} pts in Floor Plan View"
                )
            elif c_camera_points < 4:
                print(
                    f"Not enough points: Need atlest 4, but got {c_camera_points} pts"
                )
            else:
                camera_points = np.array(app_data["camera"]["points"], dtype=np.float32)
                floor_plan_points = np.array(
                    app_data["floor_plan"]["points"], dtype=np.float32
                )
                homo, _ = cv2.findHomography(camera_points, floor_plan_points)
                mean_error, errors = calculate_reprojection_error(
                    homo, camera_points, floor_plan_points
                )
                print(f"Homography matrix:\n{homo}")
                print(f"Mean reprojection error: {mean_error:.2f} px")
                print("Individual errors:", " ".join(f"{e:.2f}" for e in errors))
                if key_press == ord("s"):
                    save_data(
                        homo, camera_points, floor_plan_points, mean_error, errors
                    )
                    print("Saved Calibration")
        elif key_press in (27, ord("q")):
            break
        elif any(
            cv2.getWindowProperty(winname, cv2.WND_PROP_VISIBLE) < 1
            for winname in winnames
        ):
            break


if __name__ == "__main__":
    main()
