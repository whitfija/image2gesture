# visualizing intermediate stages of processing for debugging and reporting
import cv2
import numpy as np
import sys
import os


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import SAMPLES_DIR, SHOW_DEBUG_VIEW
from pipeline.processing.segmentation.segmentation import segment_skin
from pipeline.processing.morphology.morphology import apply_morphology
from pipeline.processing.contours.contours import extract_contour_features

def make_debug_view(
    raw: np.ndarray,
    hsv_mask: np.ndarray,
    morph_result: np.ndarray,
    contour_overlay: np.ndarray,
    label: str = ""
) -> np.ndarray:
    """
    Assembles a 2x2 debug grid:
        [raw frame]        [hsv mask]
        [morph result]     [contour overlay]
    All inputs should be BGR or will be converted.
    """
    h, w = raw.shape[:2]

    def to_bgr(img):
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img.copy()

    def labeled(img, text):
        out = to_bgr(img)
        cv2.putText(out, text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return out

    tl = labeled(raw,          "Raw Frame")
    tr = labeled(hsv_mask,     "HSV Mask")
    bl = labeled(morph_result, "Morphology")
    br = labeled(contour_overlay, "Contours")

    top = np.hstack([tl, tr])
    bot = np.hstack([bl, br])
    grid = np.vstack([top, bot])

    if label:
        cv2.putText(grid, f"Gesture: {label}",
                    (10, grid.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

    return grid


def show_debug_view(grid: np.ndarray, window_name: str = "Debug View"):
    if SHOW_DEBUG_VIEW:
        cv2.imshow(window_name, grid)


def save_debug_view(grid: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, grid)


if __name__ == "__main__":
    from config import OUTPUT_DIR

    # L, open, closed, peace, thumb
    gesture = "L"
    folder = os.path.join(SAMPLES_DIR, gesture)
    img_file = os.listdir(folder)[0]

    frame = cv2.imread(os.path.join(folder, img_file))
    frame = cv2.resize(frame, (400, 300))

    hsv, mask = segment_skin(frame)
    stages = apply_morphology(mask)
    features = extract_contour_features(stages["closed"], frame)
    contour_overlay = features["overlay"] if features["contour"] is not None else frame.copy()

    grid = make_debug_view(
        raw=frame,
        hsv_mask=mask,
        morph_result=stages["closed"],
        contour_overlay=contour_overlay,
        label=gesture
    )

    show_debug_view(grid)
    save_debug_view(grid, os.path.join(OUTPUT_DIR, f"debug_{gesture}.png"))

    cv2.waitKey(0)
    cv2.destroyAllWindows()