# pipeline/processing/segmentation/segmentation.py
import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import HSV_LOWER, HSV_UPPER, ROI_TOP, ROI_BOTTOM, ROI_LEFT, ROI_RIGHT

def apply_roi(mask: np.ndarray) -> np.ndarray:
    """Zero out everything outside the ROI."""
    h, w = mask.shape[:2]
    y1, y2 = int(h * ROI_TOP), int(h * (1 - ROI_BOTTOM))
    x1, x2 = int(w * ROI_LEFT), int(w * (1 - ROI_RIGHT))
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    roi_mask[y1:y2, x1:x2] = 255
    return cv2.bitwise_and(mask, roi_mask)

def segment_skin(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Takes a BGR frame, returns (hsv, mask).
    - Gaussian blur pre-pass to reduce noise
    - HSV color segmentation
    - ROI applied
    """
    blurred = cv2.GaussianBlur(frame, (7, 7), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    lower = np.array(HSV_LOWER)
    upper = np.array(HSV_UPPER)
    mask = cv2.inRange(hsv, lower, upper)

    mask = apply_roi(mask)

    return hsv, mask

if __name__ == "__main__":
    import os
    from config import SAMPLES_DIR

    # grabs first sample image for testing
    # L, open, closed, peace, thumb
    gesture = "thumb"
    folder = os.path.join(SAMPLES_DIR, gesture)
    img_file = os.listdir(folder)[0]
    img_path = os.path.join(folder, img_file)

    frame = cv2.imread(img_path)
    frame = cv2.resize(frame, (640, 480))

    hsv, mask = segment_skin(frame)

    out = np.hstack([
        frame,
        cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    ])
    cv2.imshow("Segmentation Test", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()