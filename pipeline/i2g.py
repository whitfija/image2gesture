"""
cli entry point for image2gesture pipeline

usage:
# static mode on test images
python i2g.py
# live webcam mode
python i2g.py --live
# run on a single image
python i2g.py --image path/to/image.jpg
# rebuild Hu templates from samples (run once after adding new samples)
python i2g.py --rebuild

"""
import cv2
import numpy as np
import argparse
import os
import sys

from config import SAMPLES_DIR, OUTPUT_DIR, SHOW_DEBUG_VIEW
from processing.segmentation.segmentation import segment_skin
from processing.morphology.morphology import apply_morphology
from processing.contours.contours import extract_contour_features
from processing.classification.classification import (
    classify_rule_based, classify_hu, get_templates
)
from debug_view import make_debug_view, show_debug_view, save_debug_view

TEST_DIR = os.path.join(os.path.dirname(__file__), "data", "test_run_1")

def run_pipeline(frame: np.ndarray, templates: dict) -> tuple[np.ndarray, dict]:
    """
    runs full processing pipeline on a single frame.
    Returns (debug_grid, results_dict)
    """
    frame = cv2.resize(frame, (400, 300))

    # 1: segmentation
    _, mask = segment_skin(frame)

    # 2: morphology
    stages = apply_morphology(mask)
    clean_mask = stages["closed"]

    # 3: contour extraction
    features = extract_contour_features(clean_mask, frame)

    # 4: classification
    if features["contour"] is None:
        label = "none"
        overlay = frame.copy()
        cv2.putText(overlay, "No hand detected", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    else:
        label, overlay = classify_combined(features, templates)

    grid = make_debug_view(
        raw=frame,
        hsv_mask=mask,
        morph_result=clean_mask,
        contour_overlay=overlay,
        label=label
    )

    results = {
        "label":        label,
        "finger_count": features.get("finger_count", "N/A"),
        "aspect_ratio": features.get("aspect_ratio", "N/A"),
    }

    return grid, results

def classify_combined(features: dict, templates: dict) -> tuple[str, np.ndarray]:
    """
    rule-based classification, tie break with Hu moments if confidence is low.
    Returns (label, overlay_with_label)
    """
    AMBIGUOUS_PAIRS = [{"thumb", "L"}, {"peace", "thumb"}]

    rule_label, rule_conf = classify_rule_based(features)
    final_label = rule_label

    for pair in AMBIGUOUS_PAIRS:
        if rule_label in pair and rule_conf < 1.0:
            hu_label, _ = classify_hu(features, templates)
            if hu_label in pair:
                final_label = hu_label
                break

    overlay = features["overlay"].copy()
    cv2.putText(overlay, f"Gesture: {final_label}", (10, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

    return final_label, overlay

# cli modes

def run_static(templates: dict, image_path: str = None):
    """
    static
    run pipeline on test images.
    if image_path provided, run on that single image.
    otherwise iterate through all images in test dir.
    """
    if image_path:
        paths = [image_path]
    else:
        exts = (".jpg", ".jpeg", ".png")
        paths = [
            os.path.join(TEST_DIR, f)
            for f in sorted(os.listdir(TEST_DIR))
            if f.lower().endswith(exts)
        ]

    if not paths:
        print(f"No images found in {TEST_DIR}")
        return

    print(f"\nRunning static pipeline on {len(paths)} image(s)...")
    print("Press any key to advance, 'q' to quit.\n")

    for path in paths:
        frame = cv2.imread(path)
        if frame is None:
            print(f"Could not load: {path}")
            continue

        grid, results = run_pipeline(frame, templates)

        print(f"[{os.path.basename(path)}]")
        print(f"  Label:        {results['label']}")
        print(f"  Fingers:      {results['finger_count']}")
        print(f"  Aspect ratio: {results['aspect_ratio']:.3f}" 
              if isinstance(results['aspect_ratio'], float) 
              else f"  Aspect ratio: {results['aspect_ratio']}")

        # save output
        out_name = f"static_{os.path.splitext(os.path.basename(path))[0]}.png"
        save_debug_view(grid, os.path.join(OUTPUT_DIR, out_name))

        show_debug_view(grid, window_name=f"i2g static: {os.path.basename(path)}")
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    print("\nstatic run complete.")


def run_live(templates: dict):
    """
    live mode
    webcam loop.
    Press 'q' to quit, 's' to save current frame to output/.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("could not open webcam.")
        return

    print("\nlive mode active. Press 'q' to quit, 's' to save frame.")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("failed to grab frame.")
            break

        grid, results = run_pipeline(frame, templates)

        show_debug_view(grid, window_name="i2g live")

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            out_path = os.path.join(OUTPUT_DIR, f"live_capture_{frame_count:04d}.png")
            save_debug_view(grid, out_path)
            print(f"Saved: {out_path}")
            frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("Live mode ended.")

# cli

def parse_args():
    parser = argparse.ArgumentParser(description="image2gesture pipeline")
    parser.add_argument("--live",    action="store_true", help="Run live webcam mode")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild Hu templates from samples")
    parser.add_argument("--image",   type=str, default=None, help="Run on a single image path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # hu templates
    templates = get_templates(force_rebuild=args.rebuild)
    if args.rebuild:
        print("templates rebuilt. exiting.")
        sys.exit(0)

    if args.live:
        run_live(templates)
    else:
        run_static(templates, image_path=args.image)