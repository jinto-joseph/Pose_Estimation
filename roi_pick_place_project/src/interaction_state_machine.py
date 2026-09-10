"""
Object-Interaction State Machine
=================================
Tracks the interaction state of a target object (default: backpack) relative
to an ROI, using two signals computed each frame from YOLO detection + YOLO-Pose:

    object_in_roi   : is the object's center inside the ROI polygon?
    hand_near_object: is a wrist keypoint close enough to the object to count
                       as touching/holding it?

States: IDLE -> PICKING -> PICKED -> PLACING -> PLACED -> (back to IDLE)

Debouncing: a condition must hold for DEBOUNCE_FRAMES consecutive frames
before the state actually changes, so single-frame detection noise doesn't
flip the state machine.

Stationarity: PLACED additionally requires the object to have stopped moving
(not just be inside the ROI) over the last STATIONARY_WINDOW frames.
"""

from collections import deque

import numpy as np


class InteractionStateMachine:
    def __init__(self, debounce_frames=5, stationary_window=8, stationary_thresh_px=6.0):
        self.state = "IDLE"
        self.debounce_frames = debounce_frames
        self.stationary_window = stationary_window
        self.stationary_thresh_px = stationary_thresh_px

        self._pending_state = None
        self._pending_count = 0
        self._center_history = deque(maxlen=stationary_window)
        self._events = []  # (frame_idx, old_state, new_state)

    def _is_stationary(self):
        if len(self._center_history) < self._center_history.maxlen:
            return False
        pts = np.array(self._center_history)
        # max pairwise spread across the window — small spread = not moving
        spread = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
        return spread < self.stationary_thresh_px

    def _propose(self, frame_idx, candidate_state):
        """Debounce: only commit a state change after it's been proposed
        for `debounce_frames` consecutive frames."""
        if candidate_state == self.state:
            self._pending_state = None
            self._pending_count = 0
            return

        if candidate_state == self._pending_state:
            self._pending_count += 1
        else:
            self._pending_state = candidate_state
            self._pending_count = 1

        if self._pending_count >= self.debounce_frames:
            old_state = self.state
            self.state = candidate_state
            self._events.append((frame_idx, old_state, candidate_state))
            self._pending_state = None
            self._pending_count = 0

    def update(self, frame_idx, object_in_roi, hand_near_object, object_center=None):
        """Call once per frame with the current frame's signals.
        object_center: (x, y) in pixels, needed only for the PLACED stationarity check.
        Returns the current committed state (str)."""

        if object_center is not None:
            self._center_history.append(object_center)

        if self.state == "IDLE":
            if object_in_roi and hand_near_object:
                self._propose(frame_idx, "PICKING")

        elif self.state == "PICKING":
            if not object_in_roi:
                self._propose(frame_idx, "PICKED")
            elif not hand_near_object:
                # hand left without the object actually leaving the ROI — false start
                self._propose(frame_idx, "IDLE")

        elif self.state == "PICKED":
            if object_in_roi and hand_near_object:
                self._propose(frame_idx, "PLACING")

        elif self.state == "PLACING":
            if not hand_near_object and object_in_roi and self._is_stationary():
                self._propose(frame_idx, "PLACED")
            """elif not object_in_roi:
                # picked back up / moved away again before settling
                self._propose(frame_idx, "PICKED")
"""
        elif self.state == "PLACED":
            if hand_near_object and object_in_roi:
                self._propose(frame_idx, "PICKING")

        return self.state

    def events(self):
        return list(self._events)
