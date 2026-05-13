"""
contour extraction and feature computation 
    for gesture classification

- find contours, keep largest (hand)
- compute convex hull (hand shape)
- compute convexity defects (finger valleys)
- filter defects by depth to remove noise
- count fingers from defects
- compute Hu moments for gesture discrimination
- aspect ratio of bounding box to separate similar finger counts    
"""

import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import MIN_CONTOUR_AREA, SAMPLES_DIR, DEFECT_MIN_DEPTH
from processing.segmentation.segmentation import segment_skin
from processing.morphology.morphology import apply_morphology

def get_largest_contour(mask: np.ndarray):
    """Find contours, return only the largest by area (that's the hand)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_CONTOUR_AREA:
        return None
    return largest

def get_convex_hull(contour):
    """Compute convex hull of the hand contour."""
    return cv2.convexHull(contour)

def get_convexity_defects(contour):
    """
    convexity defects, the gaps between fingers.
    for each defect: (start, end, far point, depth)
    far point is the fingertip valley; depth tells us defect significance.
    """
    hull_indices = cv2.convexHull(contour, returnPoints=False)
    if hull_indices is None or len(hull_indices) < 3:
        return None
    defects = cv2.convexityDefects(contour, hull_indices)
    return defects

def filter_defects(defects, contour, min_depth: float = DEFECT_MIN_DEPTH) -> list:
    """
    Filter defects by depth to remove noise.
    min_depth filters out shallow defects that aren't real finger valleys.
    Returns list of (start, end, far) point tuples.
    """
    filtered = []
    if defects is None:
        return filtered
    for defect in defects:
        start_idx, end_idx, far_idx, depth = defect[0]
        if depth > min_depth:
            start = tuple(contour[start_idx][0])
            end   = tuple(contour[end_idx][0])
            far   = tuple(contour[far_idx][0])
            filtered.append((start, end, far))
    return filtered

def count_fingers(defects: list) -> int:
    """Estimate finger count from defect count. Defects = valleys between fingers."""
    return len(defects) + 1 if defects else 1

def get_hu_moments(contour) -> np.ndarray:
    """
    Hu moments for gesture discrimination
    log-transformed
    normalized area added to separate compact shapes
    """
    moments = cv2.moments(contour)
    hu = cv2.HuMoments(moments).flatten()

    # log to bring to comparable scales & handle small values
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    # normalized area
    area = cv2.contourArea(contour)
    area_normalized = np.log10(area + 1) / 10.0
    hu = np.append(hu, area_normalized)
    
    return hu

def draw_contour_overlay(frame: np.ndarray, contour, hull, defects: list) -> np.ndarray:
    """Draw contour, hull, and defect points onto a copy of frame."""
    overlay = frame.copy()
    if contour is None:
        return overlay

    # GREEN contour
    cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 2)
    # BLUE convex hull
    cv2.drawContours(overlay, [hull], -1, (255, 0, 0), 2)

    # defect points - RED far points (valleys), YELLOW start/end (fingertips)
    for start, end, far in defects:
        cv2.circle(overlay, far,   6, (0, 0, 255), -1)    # valley
        cv2.circle(overlay, start, 6, (0, 255, 255), -1)  # fingertip
        cv2.circle(overlay, end,   6, (0, 255, 255), -1)  # fingertip

    # finger count
    finger_count = count_fingers(defects)
    cv2.putText(overlay, f"Fingers: {finger_count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return overlay

def extract_contour_features(mask: np.ndarray, frame: np.ndarray) -> dict:
    """
    full contour pipeline, returns all needed features
    """
    contour = get_largest_contour(mask)
    if contour is None:
        return {"contour": None}

    hull        = get_convex_hull(contour)
    raw_defects = get_convexity_defects(contour)
    defects     = filter_defects(raw_defects, contour)
    finger_count= count_fingers(defects)
    hu_moments  = get_hu_moments(contour)
    overlay     = draw_contour_overlay(frame, contour, hull, defects)

    # aspect ratio, bounding box of contour
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h if h > 0 else 0

    return {
        "contour":      contour,
        "hull":         hull,
        "defects":      defects,
        "finger_count": finger_count,
        "hu_moments":   hu_moments,
        "aspect_ratio": aspect_ratio,
        "overlay":      overlay,
    }


if __name__ == "__main__":
    from config import OUTPUT_DIR
    from debug_view import make_debug_view, show_debug_view, save_debug_view

    # L, open, closed, peace, thumb
    gesture = "closed"
    folder  = os.path.join(SAMPLES_DIR, gesture)
    img_file = os.listdir(folder)[0]

    frame = cv2.imread(os.path.join(folder, img_file))
    frame = cv2.resize(frame, (400, 300))

    _, mask   = segment_skin(frame)
    stages    = apply_morphology(mask)
    clean_mask = stages["closed"]

    features  = extract_contour_features(clean_mask, frame)

    if features["contour"] is None:
        print("no contour found, mask too fragmented or hand not detected")
    else:
        print(f"Finger count:  {features['finger_count']}")
        print(f"Aspect ratio:  {features['aspect_ratio']:.3f}")
        print(f"Hu moments:    {features['hu_moments']}")

        grid = make_debug_view(
            raw=frame,
            hsv_mask=mask,
            morph_result=clean_mask,
            contour_overlay=features["overlay"],
            label=gesture
        )

        show_debug_view(grid)
        save_debug_view(grid, os.path.join(OUTPUT_DIR, f"debug_{gesture}.png"))

    cv2.waitKey(0)
    cv2.destroyAllWindows()