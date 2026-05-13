import cv2
import numpy as np
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import SAMPLES_DIR, GESTURE_LABELS, DATA_DIR
from processing.segmentation.segmentation import segment_skin
from processing.morphology.morphology import apply_morphology
from processing.contours.contours import extract_contour_features

TEMPLATES_PATH = os.path.join(DATA_DIR, "hu_templates.json")

# rule-based thresholds
FINGER_COUNT_MAP = {
    "open":   (4, 5),
    "peace":  (2, 3),
    "thumb":  (0, 1),
    "L":      (2, 2),
    "closed": (0, 1),
}

# aspect ratio ranges (width/height) per gesture
ASPECT_RATIO_MAP = {
    "open":   (0.6, 1.0),
    "peace":  (0.2, 0.5),
    "thumb":  (0.3, 0.6),
    "L":      (0.5, 0.9),
    "closed": (0.6, 1.1),
}

def classify_rule_based(features: dict) -> tuple[str, float]:
    """
    Rule-based classifier using finger count + aspect ratio.
    Returns (gesture_label, confidence).
    Confidence is a simple score - number of rules matched / total rules.
    """
    if features.get("contour") is None:
        return "none", 0.0

    finger_count = features["finger_count"]
    aspect_ratio = features["aspect_ratio"]
    contour_area = cv2.contourArea(features["contour"])

    scores = {}
    for gesture in GESTURE_LABELS:
        score = 0
        total = 2

        # Finger count match
        lo, hi = FINGER_COUNT_MAP[gesture]
        if lo <= finger_count <= hi:
            score += 1

        # Aspect ratio match
        alo, ahi = ASPECT_RATIO_MAP[gesture]
        if alo <= aspect_ratio <= ahi:
            score += 1

        scores[gesture] = score / total

    best = max(scores, key=scores.get)
    return best, scores[best]

def classify_hu(features: dict, templates: dict) -> tuple[str, float]:
    """
    Nearest-neighbor classifier on Hu moment vectors
    templates: {gesture_label: mean_hu_vector} - built from sample images.
    Returns (gesture_label, distance) - lower distance = more confident.
    """
    if features.get("contour") is None or not templates:
        return "none", float("inf")

    hu = features["hu_moments"]
    best_label = "none"
    best_dist  = float("inf")

    for label, template_hu in templates.items():
        dist = np.linalg.norm(hu - template_hu)
        if dist < best_dist:
            best_dist  = dist
            best_label = label

    return best_label, best_dist

def build_hu_templates(samples_dir: str) -> dict:
    """
    build mean Hu moment vector per gesture from sample images.
    loads all samples and averages their Hu vectors.
    """
    templates = {}
    for gesture in GESTURE_LABELS:
        folder = os.path.join(samples_dir, gesture)
        if not os.path.exists(folder):
            continue

        hu_vectors = []
        for fname in os.listdir(folder):
            if not fname.lower().endswith((".jpg", ".png")):
                continue
            path  = os.path.join(folder, fname)
            frame = cv2.imread(path)
            if frame is None:
                continue
            frame = cv2.resize(frame, (640, 480))
            _, mask   = segment_skin(frame)
            stages    = apply_morphology(mask)
            features  = extract_contour_features(stages["closed"], frame)
            if features.get("hu_moments") is not None:
                hu_vectors.append(features["hu_moments"])

        if hu_vectors:
            templates[gesture] = np.mean(hu_vectors, axis=0)
            print(f"Built template for '{gesture}' from {len(hu_vectors)} samples")

    return templates

def save_templates(templates: dict):
    serializable = {k: v.tolist() for k, v in templates.items()}
    with open(TEMPLATES_PATH, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"Templates saved to {TEMPLATES_PATH}")

def load_templates() -> dict | None:
    if not os.path.exists(TEMPLATES_PATH):
        return None
    with open(TEMPLATES_PATH, 'r') as f:
        raw = json.load(f)
    return {k: np.array(v) for k, v in raw.items()}

def get_templates(force_rebuild: bool = False) -> dict:
    if not force_rebuild:
        templates = load_templates()
        if templates:
            print("Loaded templates from cache")
            return templates
    print("Building templates from samples...")
    templates = build_hu_templates(SAMPLES_DIR)
    save_templates(templates)
    return templates

if __name__ == "__main__":
    from config import OUTPUT_DIR
    from debug_view import make_debug_view, show_debug_view, save_debug_view

    print("Building Hu moment templates from samples...")
    templates = build_hu_templates(SAMPLES_DIR)

    # testing one image per gesture
    for gesture in GESTURE_LABELS:
        folder = os.path.join(SAMPLES_DIR, gesture)
        if not os.path.exists(folder):
            continue

        img_file = os.listdir(folder)[0]
        frame    = cv2.imread(os.path.join(folder, img_file))
        frame    = cv2.resize(frame, (640, 480))

        _, mask    = segment_skin(frame)
        stages     = apply_morphology(mask)
        clean_mask = stages["closed"]
        features   = extract_contour_features(clean_mask, frame)

        rule_label, rule_conf = classify_rule_based(features)
        hu_label,   hu_dist   = classify_hu(features, templates)

        print(f"\n[{gesture}]")
        print(f"  Rule-based:  {rule_label} (conf: {rule_conf:.2f})")
        print(f"  Hu moments:  {hu_label}   (dist: {hu_dist:.4f})")
        print(f"  Fingers:     {features.get('finger_count', 'N/A')}")
        print(f"  Aspect:      {features.get('aspect_ratio', 'N/A'):.3f}" 
              if features.get('aspect_ratio') else "  Aspect: N/A")

        if features.get("overlay") is not None:
            grid = make_debug_view(
                raw=frame,
                hsv_mask=mask,
                morph_result=clean_mask,
                contour_overlay=features["overlay"],
                label=f"true={gesture} | rule={rule_label} | hu={hu_label}"
            )
            save_debug_view(grid, os.path.join(OUTPUT_DIR, f"classify_{gesture}.png"))