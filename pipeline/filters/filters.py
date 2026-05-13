"""
gesture-driven image filters

GESTURE   FILTER
open      Gaussian blur
closed    Histogram equalization
peace     Ordered dithering (Bayer 4x4)
thumb     Negative
L         Halftone dots
"""

import cv2
import numpy as np


def filter_gaussian_blur(frame: np.ndarray) -> np.ndarray:
    """
    open
    gaussian blur (spatial domain smoothing).
    """
    return cv2.GaussianBlur(frame, (21, 21), 0)


def filter_histogram_eq(frame: np.ndarray) -> np.ndarray:
    """
    closed
    histogram equalization
    applied to luminance (YCrCb) to avoid color distortion
    """
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def filter_ordered_dither(frame: np.ndarray) -> np.ndarray:
    """
    peace
    ordered dithering using a 4x4 Bayer matrix
    - converts to grayscale, 
    - applies Bayer threshold map 
    - outputs a 2-tone (black/white) BGR result
    """
    # standard 4x4 Bayer threshold matrix, normalized to 0-255
    bayer_4x4 = np.array([
        [  0, 136,  34, 170],
        [204,  68, 238, 102],
        [ 51, 187,  17, 153],
        [255, 119, 221,  85],
    ], dtype=np.float32)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape

    # tile Bayer matrix to frame size
    tile_h = int(np.ceil(h / 4))
    tile_w = int(np.ceil(w / 4))
    threshold_map = np.tile(bayer_4x4, (tile_h, tile_w))[:h, :w]

    dithered = np.where(gray > threshold_map, 255, 0).astype(np.uint8)
    return cv2.cvtColor(dithered, cv2.COLOR_GRAY2BGR)


def filter_negative(frame: np.ndarray) -> np.ndarray:
    """
    thumb
    photographic negative.
    """
    return cv2.bitwise_not(frame)


def filter_halftone(frame: np.ndarray, cell_size: int = 10) -> np.ndarray:
    """
    L 
    halftone dot simulation.
    - divides the frame into grid cells, draws a filled circle in each
    - cell whose radius is proportional to the cell's mean luminance

    cell_size: pixel size of each halftone cell (bigger = coarser dots)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # white canvas
    canvas = np.ones((h, w), dtype=np.uint8) * 255

    for y in range(0, h, cell_size):
        for x in range(0, w, cell_size):
            cell = gray[y:y + cell_size, x:x + cell_size]
            mean_val = cell.mean()

            # darker cell = larger dot
            # radius scales from 0 (white) to cell_size/2 (full black)
            radius = int((1.0 - mean_val / 255.0) * (cell_size / 2))

            cx = x + cell_size // 2
            cy = y + cell_size // 2
            if radius > 0:
                cv2.circle(canvas, (cx, cy), radius, 0, -1)

    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


def filter_passthrough(frame: np.ndarray) -> np.ndarray:
    """no gesture, no filter"""
    return frame.copy()


FILTER_MAP = {
    "open":   filter_gaussian_blur,
    "closed": filter_histogram_eq,
    "peace":  filter_ordered_dither,
    "thumb":  filter_negative,
    "L":      filter_halftone,
    "none":   filter_passthrough,
}

FILTER_LABELS = {
    "open":   "gaussian blur",
    "closed": "histogram equalization",
    "peace":  "ordered dithering",
    "thumb":  "negative",
    "L":      "halftone dots",
    "none":   "no filter",
}


def apply_filter(frame: np.ndarray, gesture_label: str) -> np.ndarray:
    """
    apply the filter for the detected gesture
    Returns labeled frame
    """
    fn = FILTER_MAP.get(gesture_label, filter_passthrough)
    filtered = fn(frame)

    filter_name = FILTER_LABELS.get(gesture_label, "")
    cv2.putText(filtered, f"{gesture_label} - {filter_name}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    return filtered