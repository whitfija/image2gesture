import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import MORPH_KERNEL_SIZE, MORPH_CLOSE_KERNEL_SIZE, SAMPLES_DIR
from processing.segmentation.segmentation import segment_skin

def apply_morphology(mask: np.ndarray) -> dict[str, np.ndarray]:
    """
    morphological operations to clean up the mask
    
    1. opening (erode, then dilate) - removes speckle noise
    2. closing (dilatem then erode) - fills interior holes in hand blob

    returns intermediate stages for debug view.

    2 kernels: small for opening, large for closing to fill bigger holes
    configure small kernel in config.py
    """
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_CLOSE_KERNEL_SIZE)

    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_large)

    return {
        "raw_mask":  mask,
        "opened":    opened,
        "closed":    closed,
    }

if __name__ == "__main__":
    # L, open, closed, peace, thumb
    gesture = "thumb"
    folder = os.path.join(SAMPLES_DIR, gesture)
    img_file = os.listdir(folder)[1]

    frame = cv2.imread(os.path.join(folder, img_file))
    frame = cv2.resize(frame, (400, 300))

    _, mask = segment_skin(frame)
    stages = apply_morphology(mask)

    raw    = cv2.cvtColor(stages["raw_mask"], cv2.COLOR_GRAY2BGR)
    opened = cv2.cvtColor(stages["opened"],   cv2.COLOR_GRAY2BGR)
    closed = cv2.cvtColor(stages["closed"],   cv2.COLOR_GRAY2BGR)

    cv2.putText(raw,    "Raw Mask", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.putText(opened, "Opened",   (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.putText(closed, "Closed",   (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    out = np.hstack([raw, opened, closed])
    cv2.imshow("Morphology Stages", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()