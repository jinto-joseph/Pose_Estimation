"""
Generic Object Pick/Place Detection Pipeline
=============================================

Combines:
  - YOLO detection + tracking for person and target objects
  - YOLO-Pose for complete human pose/keypoints
  - Auto-initialized, frozen ROI from the object's resting bbox
  - Hand-to-object interaction detection
  - Temporal state machine:
        IDLE -> PICKING -> PICKED -> PLACING -> PLACED

Currently configured for multiple COCO object classes.

Run:
    python roi_pick_place.py \
        --video ../input/input.mp4 \
        --output ../output/annotated.mp4
"""

import argparse

import cv2
import numpy as np
from ultralytics import YOLO

from interaction_state_machine import InteractionStateMachine
from roi_utils import ROIInitializer


# ========================================================
# Configuration
# ========================================================

TARGET_OBJECT_CLASSES = [
    "backpack",
    "bottle",
    "cell phone",
    "book",
    "cup",
    "laptop",
    "handbag",
    "suitcase",
]

PERSON_CLASS_ID = 0

# COCO pose keypoint indices
LEFT_WRIST = 9
RIGHT_WRIST = 10

# Hand-to-object distance as fraction of object bbox diagonal
HAND_NEAR_FRACTION = 0.6

# State-machine parameters
DEBOUNCE_FRAMES = 5
STATIONARY_WINDOW = 8
STATIONARY_THRESH_PX = 6.0

# ROI initialization
ROI_STABILITY_FRAMES = 8
ROI_EXPAND_SCALE = 1
ROI_MIN_MARGIN_PX = 20


# ========================================================
# Argument parsing
# ========================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video",
        required=True,
        help="Input video path"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output annotated video path"
    )

    parser.add_argument(
        "--det-model",
        default="yolo11l.pt",
        help="YOLO detection model"
    )

    parser.add_argument(
        "--pose-model",
        default="yolo11l-pose.pt",
        help="YOLO pose model"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Detection confidence threshold"
    )

    return parser.parse_args()


# ========================================================
# Geometry utilities
# ========================================================

def point_in_rect(point, rect_xyxy):
    """Check whether a point lies inside a rectangular ROI."""

    x, y = point
    x1, y1, x2, y2 = rect_xyxy

    return x1 <= x <= x2 and y1 <= y <= y2


def bbox_center(xyxy):
    """Return center point of bounding box."""

    x1, y1, x2, y2 = xyxy

    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2
    )


def bbox_diagonal(xyxy):
    """Return diagonal length of bounding box."""

    x1, y1, x2, y2 = xyxy

    return float(
        np.hypot(
            x2 - x1,
            y2 - y1
        )
    )


def dist_point_to_bbox(point, xyxy):
    """
    Distance from a point to the nearest edge of a bounding box.

    Returns 0 if the point is inside the bbox.
    """

    px, py = point

    x1, y1, x2, y2 = xyxy

    dx = max(
        x1 - px,
        0,
        px - x2
    )

    dy = max(
        y1 - py,
        0,
        py - y2
    )

    return float(
        np.hypot(dx, dy)
    )


# ========================================================
# Main
# ========================================================

def main():

    args = parse_args()

    # ----------------------------------------------------
    # Load models
    # ----------------------------------------------------

    print("Loading YOLO detection model...")
    det_model = YOLO(args.det_model)

    print("Loading YOLO pose model...")
    pose_model = YOLO(args.pose_model)

    class_names = det_model.names

    # ----------------------------------------------------
    # Find target class IDs
    # ----------------------------------------------------

    target_class_ids = []

    for cid, name in class_names.items():

        if name in TARGET_OBJECT_CLASSES:
            target_class_ids.append(cid)

    if not target_class_ids:

        raise ValueError(
            f"None of {TARGET_OBJECT_CLASSES} found "
            f"in detection model classes."
        )

    print("\nTarget classes:")

    for cid in target_class_ids:

        print(
            f"  {cid}: {class_names[cid]}"
        )

    # ----------------------------------------------------
    # Open video
    # ----------------------------------------------------

    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video: {args.video}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    ) or 25.0

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    print(
        f"\nVideo: {width}x{height} @ {fps:.2f} FPS"
    )

    # ----------------------------------------------------
    # Output video
    # ----------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        args.output,
        fourcc,
        fps,
        (width, height)
    )

    # ----------------------------------------------------
    # State machine
    # ----------------------------------------------------

    sm = InteractionStateMachine(
        debounce_frames=DEBOUNCE_FRAMES,
        stationary_window=STATIONARY_WINDOW,
        stationary_thresh_px=STATIONARY_THRESH_PX,
    )

    # ----------------------------------------------------
    # ROI initializer
    # ----------------------------------------------------

    roi_init = ROIInitializer(
        stability_frames=ROI_STABILITY_FRAMES,
        scale=ROI_EXPAND_SCALE,
        min_margin_px=ROI_MIN_MARGIN_PX,
    )

    frozen_roi = None

    frame_idx = 0

    # ====================================================
    # Frame loop
    # ====================================================

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # ------------------------------------------------
        # YOLO object detection + tracking
        # ------------------------------------------------

        det_results = det_model.track(
            frame,
            persist=True,
            classes=[
                PERSON_CLASS_ID,
                *target_class_ids
            ],
            conf=args.conf,
            verbose=False,
        )[0]

        # ------------------------------------------------
        # YOLO Pose
        # ------------------------------------------------

        pose_results = pose_model.predict(
            frame,
            conf=args.conf,
            verbose=False
        )[0]

        # Plot COMPLETE human pose
        annotated_frame = pose_results.plot()

        # ------------------------------------------------
        # Find target object
        # ------------------------------------------------

        object_box = None
        object_class_name = None
        best_conf = -1.0

        if det_results.boxes is not None:

            for box in det_results.boxes:

                cls_id = int(
                    box.cls[0]
                )

                conf = float(
                    box.conf[0]
                )

                if (
                    cls_id in target_class_ids
                    and conf > best_conf
                ):

                    best_conf = conf

                    object_box = (
                        box.xyxy[0]
                        .cpu()
                        .numpy()
                    )

                    object_class_name = (
                        class_names[cls_id]
                    )

        # ------------------------------------------------
        # Find nearest wrist to object
        # ------------------------------------------------

        nearest_wrist = None
        hand_near = False

        if (
            object_box is not None
            and pose_results.keypoints is not None
        ):

            min_dist = float("inf")

            for kpts in pose_results.keypoints.xy:

                kpts = (
                    kpts
                    .cpu()
                    .numpy()
                )

                for wrist_idx in (
                    LEFT_WRIST,
                    RIGHT_WRIST
                ):

                    if wrist_idx >= len(kpts):
                        continue

                    wx, wy = kpts[wrist_idx]

                    # Invalid / missing keypoint
                    if wx == 0 and wy == 0:
                        continue

                    distance = dist_point_to_bbox(
                        (wx, wy),
                        object_box
                    )

                    if distance < min_dist:

                        min_dist = distance

                        nearest_wrist = (
                            wx,
                            wy
                        )

            if nearest_wrist is not None:

                threshold = (
                    HAND_NEAR_FRACTION
                    * bbox_diagonal(object_box)
                )

                hand_near = (
                    min_dist < threshold
                )

        # ------------------------------------------------
        # Initialize ROI
        # ------------------------------------------------

        if frozen_roi is None:

            roi_init.update(
                object_box,
                hand_near
            )

            if roi_init.ready():

                frozen_roi = (
                    roi_init.get_roi()
                )

                print(
                    f"\n[frame {frame_idx}] "
                    f"ROI frozen at "
                    f"{frozen_roi}"
                )

                print(
                    f"Object: "
                    f"{object_class_name}"
                )

        # ------------------------------------------------
        # Determine object position relative to ROI
        # ------------------------------------------------

        object_in_roi = False
        object_center = None

        if object_box is not None:

            object_center = bbox_center(
                object_box
            )

            if frozen_roi is not None:

                object_in_roi = point_in_rect(
                    object_center,
                    frozen_roi
                )

        # ------------------------------------------------
        # State machine
        # ------------------------------------------------

        if frozen_roi is not None:

            state = sm.update(
                frame_idx,
                object_in_roi,
                hand_near,
                object_center
            )

        else:

            state = "INITIALIZING_ROI"

        # =================================================
        # Visualization
        # =================================================

        # -------------------------------------------------
        # Draw frozen ROI
        # -------------------------------------------------

        if frozen_roi is not None:

            rx1, ry1, rx2, ry2 = [
                int(v)
                for v in frozen_roi
            ]

            cv2.rectangle(
                annotated_frame,
                (rx1, ry1),
                (rx2, ry2),
                (255, 0, 0),
                2
            )

            cv2.putText(
                annotated_frame,
                "REFERENCE ROI",
                (
                    rx1,
                    max(ry1 - 10, 20)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

        # -------------------------------------------------
        # Draw target object
        # -------------------------------------------------

        if object_box is not None:

            x1, y1, x2, y2 = (
                object_box.astype(int)
            )

            if object_in_roi:

                object_color = (
                    0,
                    255,
                    0
                )

            else:

                object_color = (
                    0,
                    165,
                    255
                )

            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                object_color,
                2
            )

            label = (
                f"{object_class_name} "
                f"{best_conf:.2f}"
            )

            cv2.putText(
                annotated_frame,
                label,
                (
                    x1,
                    max(y1 - 10, 20)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                object_color,
                2
            )

        # -------------------------------------------------
        # Draw nearest wrist
        # -------------------------------------------------

        if nearest_wrist is not None:

            cv2.circle(
                annotated_frame,
                (
                    int(nearest_wrist[0]),
                    int(nearest_wrist[1])
                ),
                6,
                (0, 0, 255),
                -1
            )

        # -------------------------------------------------
        # Status
        # -------------------------------------------------

        cv2.putText(
            annotated_frame,
            f"State: {state}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2
        )

        # -------------------------------------------------
        # Frame number
        # -------------------------------------------------

        cv2.putText(
            annotated_frame,
            f"Frame: {frame_idx}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # -------------------------------------------------
        # Write frame
        # -------------------------------------------------

        writer.write(
            annotated_frame
        )

        frame_idx += 1

    # ====================================================
    # Cleanup
    # ====================================================

    cap.release()
    writer.release()

    print("\n================================")
    print("Processing complete")
    print("================================")

    print("\nState transitions:")

    for frame, old, new in sm.events():

        print(
            f"  frame {frame}: "
            f"{old} -> {new}"
        )

    print(
        f"\nOutput saved to: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()