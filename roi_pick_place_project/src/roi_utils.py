"""
ROI initialization utilities.

Instead of trusting frame 0's detection blindly (which could catch the object
mid-interaction — hand covering it, motion blur while being lifted), this waits
for a short window where the object is detected CONSISTENTLY and with NO hand
nearby (i.e. genuinely at rest), then freezes the ROI from the median box in
that window. This avoids freezing a distorted reference ROI from a bad frame.
"""

from collections import deque

import numpy as np


def expand_bbox(xyxy, scale=0.5, min_margin_px=20):
    """Expand a bbox by `scale` fraction of its own width/height, with a floor
    of `min_margin_px` so tiny/far-away objects still get a usable margin."""
    x1, y1, x2, y2 = xyxy
    width, height = x2 - x1, y2 - y1
    margin_x = max(width * scale, min_margin_px)
    margin_y = max(height * scale, min_margin_px)
    return (x1 - margin_x, y1 - margin_y, x2 + margin_x, y2 + margin_y)


class ROIInitializer:
    """Feed it (object_bbox_or_None, hand_near_bool) every frame until it
    reports ready() — then call get_roi() once for the frozen, expanded ROI."""

    def __init__(self, stability_frames=8, scale=0.5, min_margin_px=20):
        self.stability_frames = stability_frames
        self.scale = scale
        self.min_margin_px = min_margin_px
        self._window = deque(maxlen=stability_frames)
        self._frozen_roi = None

    def update(self, object_bbox, hand_near):
        if self._frozen_roi is not None:
            return  # already frozen, nothing more to do

        if object_bbox is not None and not hand_near:
            self._window.append(object_bbox)
        else:
            self._window.clear()  # any occlusion/hand-interaction resets the stability count

        if len(self._window) == self.stability_frames:
            boxes = np.array(self._window)
            median_box = tuple(np.median(boxes, axis=0))
            self._frozen_roi_source_box = median_box
            self._frozen_roi = expand_bbox(median_box, self.scale, self.min_margin_px)

    def ready(self):
        return self._frozen_roi is not None

    def get_roi(self):
        if self._frozen_roi is None:
            raise RuntimeError("ROI not yet stable — check ready() before calling get_roi().")
        return self._frozen_roi
