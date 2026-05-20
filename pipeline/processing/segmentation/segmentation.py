# pipeline/processing/segmentation/segmentation.py
import cv2
import numpy as np
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import HSV_LOWER, HSV_UPPER, ROI_TOP, ROI_BOTTOM, ROI_LEFT, ROI_RIGHT, DATA_DIR

HSV_CONFIG_PATH = os.path.join(DATA_DIR, "hsv_config.json")

def save_hsv_config(lower: tuple, upper: tuple):
    data = {"lower": list(lower), "upper": list(upper)}
    os.makedirs(os.path.dirname(HSV_CONFIG_PATH), exist_ok=True)
    with open(HSV_CONFIG_PATH, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"HSV config saved to {HSV_CONFIG_PATH}")

def load_hsv_config() -> tuple[tuple, tuple] | None:
    if not os.path.exists(HSV_CONFIG_PATH):
        return None
    with open(HSV_CONFIG_PATH, 'r') as f:
        data = json.load(f)
    lower = tuple(data["lower"])
    upper = tuple(data["upper"])
    # print("Loaded HSV config from hsv_config.json")
    return lower, upper

def get_hsv_range() -> tuple[tuple, tuple]:
    """prioritize cached HSV values, otherwise use defaults."""
    cached = load_hsv_config()
    if cached:
        return cached
    return HSV_LOWER, HSV_UPPER

def apply_roi(mask: np.ndarray) -> np.ndarray:
    """zero everything outside the region of interest"""
    h, w = mask.shape[:2]
    y1, y2 = int(h * ROI_TOP), int(h * (1 - ROI_BOTTOM))
    x1, x2 = int(w * ROI_LEFT), int(w * (1 - ROI_RIGHT))
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    roi_mask[y1:y2, x1:x2] = 255
    return cv2.bitwise_and(mask, roi_mask)

def segment_skin(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    takes BGR frame, returns (hsv, mask).
    - Gaussian blur pre-pass to reduce noise
    - HSV color segmentation
    - ROI applied
    """
    lower, upper = get_hsv_range()
    blurred = cv2.GaussianBlur(frame, (7, 7), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
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
    img_file = os.listdir(folder)[1]
    img_path = os.path.join(folder, img_file)
    print(f"Testing segmentation on {img_path}")

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