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

# eval mode
# static on data/eval/ with ground truth labels as folder names
# expects: data/eval/open/, data/eval/closed/, etc.
python i2g.py --eval

"""
import cv2
import numpy as np
import argparse
import os
import sys

from config import SAMPLES_DIR, OUTPUT_DIR, DATA_DIR, SHOW_DEBUG_VIEW, HSV_LOWER, HSV_UPPER
from processing.segmentation.segmentation import segment_skin
from processing.morphology.morphology import apply_morphology
from processing.contours.contours import extract_contour_features
from processing.classification.classification import (
    classify_rule_based, classify_hu, get_templates
)
from debug_view import make_debug_view, show_debug_view, save_debug_view

TEST_DIR = os.path.join(DATA_DIR, "input")

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
    AMBIGUOUS_PAIRS = [{"thumb", "L"}, {"peace", "thumb"}, {"peace", "L"}]

    rule_label, rule_conf = classify_rule_based(features)
    final_label = rule_label

    for pair in AMBIGUOUS_PAIRS:
        if rule_label in pair and rule_conf < 0.75:
            hu_label, _ = classify_hu(features, templates)
            # tiebreakers
            if rule_conf <= 0.5:
                # rules are split, Hu decides
                final_label = hu_label
            elif hu_label in pair:
                # rules lean one way, Hu confirms or flips within pair
                final_label = hu_label
            break


    overlay = features["overlay"].copy()
    cv2.putText(overlay, f"Gesture: {final_label}", (10, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

    return final_label, overlay

def destroy_if_exists(window_name):
    try:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 0:
            cv2.destroyWindow(window_name)
    except:
        pass

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

    controls:
      'd' - debug view only
      'f' - filter view only
      'b' - both side by side
      's' - save current frame(s) to output/
      'q' - quit

    """
    from filters.filters import apply_filter

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("could not open webcam.")
        return

    view_mode = 'd'
    print("\nlive mode active.")
    print("  'd' debug view | 'f' filter view | 'b' both | 's' save | 'q' quit")
    
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("failed to grab frame.")
            break

        grid, results = run_pipeline(frame, templates)
        label = results["label"]

        raw_resized = cv2.resize(frame, (400, 300))
        filter_view = apply_filter(raw_resized, label)

        # view toggle
        if view_mode == 'd':
            show_debug_view(grid, window_name="i2g | debug")
            destroy_if_exists("i2g | filter")
        elif view_mode == 'f':
            cv2.imshow("i2g | filter", filter_view)
            destroy_if_exists("i2g | debug")
        elif view_mode == 'b':
            show_debug_view(grid, window_name="i2g | debug")
            cv2.imshow("i2g | filter", filter_view)

        # controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            view_mode = 'd'
            print("view: debug")
        elif key == ord('f'):
            view_mode = 'f'
            print("view: filter")
        elif key == ord('b'):
            view_mode = 'b'
            print("view: both")
        elif key == ord('s'):
            if view_mode in ('d', 'b'):
                out_path = os.path.join(OUTPUT_DIR, f"live_debug_{frame_count:04d}.png")
                save_debug_view(grid, out_path)
                print(f"Saved debug: {out_path}")
            if view_mode in ('f', 'b'):
                out_path = os.path.join(OUTPUT_DIR, f"live_filter_{frame_count:04d}.png")
                cv2.imwrite(out_path, filter_view)
                print(f"Saved filter: {out_path}")
            frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("live mode ended.")

def run_calibrate():
    """
    live calibration mode
    'p' - pause/unpause feed
    's' - save current HSV values to hsv_config.json
    'q' - quit calibration
    """
    from processing.segmentation.segmentation import save_hsv_config, load_hsv_config, get_hsv_range

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("could not open webcam.")
        return

    def nothing(x): pass

    cached = load_hsv_config()
    if cached:
        current_lower, current_upper = cached
        print("Starting from saved HSV calibration.")
    else:
        current_lower, current_upper = HSV_LOWER, HSV_UPPER
        print("No saved calibration found, starting from config defaults.")

    cv2.namedWindow("Calibrate")
    cv2.createTrackbar("H_min", "Calibrate", current_lower[0], 179, nothing)
    cv2.createTrackbar("H_max", "Calibrate", current_upper[0], 179, nothing)
    cv2.createTrackbar("S_min", "Calibrate", current_lower[1], 255, nothing)
    cv2.createTrackbar("S_max", "Calibrate", current_upper[1], 255, nothing)
    cv2.createTrackbar("V_min", "Calibrate", current_lower[2], 255, nothing)
    cv2.createTrackbar("V_max", "Calibrate", current_upper[2], 255, nothing)

    paused = False
    frozen = None
    print("\nCalibration mode. 'p' pause, 's' save, 'q' quit.")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640, 480))
            frozen = frame.copy()
        else:
            frame = frozen.copy()

        h1 = cv2.getTrackbarPos("H_min", "Calibrate")
        h2 = cv2.getTrackbarPos("H_max", "Calibrate")
        s1 = cv2.getTrackbarPos("S_min", "Calibrate")
        s2 = cv2.getTrackbarPos("S_max", "Calibrate")
        v1 = cv2.getTrackbarPos("V_min", "Calibrate")
        v2 = cv2.getTrackbarPos("V_max", "Calibrate")

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([h1,s1,v1]), np.array([h2,s2,v2]))
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        status = "PAUSED - tuning" if paused else "LIVE - press 'p' to pause"
        cv2.putText(frame, status, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        combined = np.hstack([frame, mask_bgr])
        cv2.imshow("Calibrate", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'):
            paused = not paused
            print("Paused." if paused else "Resumed.")
        elif key == ord('s'):
            lower = (h1, s1, v1)
            upper = (h2, s2, v2)
            save_hsv_config(lower, upper)
            print(f"Saved: lower={lower} upper={upper}")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def run_eval(templates: dict):
    """
    eval mode
    expects data/eval/<gesture>/ folders where folder name = ground truth label.
    computes per-gesture accuracy and prints a confusion matrix.
    """
    from sklearn.metrics import classification_report, confusion_matrix

    EVAL_DIR = os.path.join(DATA_DIR, "eval")
    GESTURE_LABELS = ["open", "closed", "peace", "thumb", "L"]
    EXTS = (".jpg", ".jpeg", ".png")

    if not os.path.isdir(EVAL_DIR):
        print(f"eval directory not found: {EVAL_DIR}")
        print("create data/eval/ with subfolders named by gesture label.")
        return

    print(f"\nrunning evaluation on {EVAL_DIR}...")

    y_true, y_pred = [], []
    per_gesture = {g: {"correct": 0, "total": 0} for g in GESTURE_LABELS}

    for true_label in GESTURE_LABELS:
        folder = os.path.join(EVAL_DIR, true_label)
        if not os.path.isdir(folder):
            print(f"  [skip] no folder for '{true_label}'")
            continue

        paths = [
            os.path.join(folder, f)
            for f in sorted(os.listdir(folder))
            if f.lower().endswith(EXTS)
        ]
        if not paths:
            print(f"  [skip] no images in {folder}")
            continue

        print(f"\n[{true_label}] — {len(paths)} images")
        for path in paths:
            frame = cv2.imread(path)
            if frame is None:
                print(f"  could not load: {path}")
                continue

            _, results = run_pipeline(frame, templates)
            pred_label = results["label"]

            y_true.append(true_label)
            y_pred.append(pred_label)
            per_gesture[true_label]["total"] += 1
            if pred_label == true_label:
                per_gesture[true_label]["correct"] += 1

            marker = "CORRECT" if pred_label == true_label else f"INCORRECT (guessed: {pred_label})"
            print(f"  {os.path.basename(path):<30} {marker}")

    if not y_true:
        print("No images evaluated.")
        return

    # per-gesture accuracy summary
    print("\n--- per-gesture accuracy ---")
    for g in GESTURE_LABELS:
        t = per_gesture[g]["total"]
        if t == 0:
            continue
        c = per_gesture[g]["correct"]
        print(f"  {g:<10} {c}/{t}  ({100*c/t:.0f}%)")

    overall = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    print(f"\n  overall: {overall}/{len(y_true)}  ({100*overall/len(y_true):.0f}%)")

    # confusion matrix
    print("\n--- confusion matrix ---")
    labels_present = sorted(set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels_present)
    header = f"{'':>10}" + "".join(f"{l:>10}" for l in labels_present)
    print(f"{header}  (predicted)")
    for i, row_label in enumerate(labels_present):
        row = "".join(f"{v:>10}" for v in cm[i])
        print(f"{row_label:>10}{row}")

    # sklearn report
    print("\n--- classification report ---")
    print(classification_report(y_true, y_pred, labels=labels_present, zero_division=0))

    # save results to output/
    out_path = os.path.join(OUTPUT_DIR, "eval_results.txt")
    with open(out_path, "w") as f:
        f.write("per-gesture accuracy\n")
        for g in GESTURE_LABELS:
            t = per_gesture[g]["total"]
            if t == 0: continue
            c = per_gesture[g]["correct"]
            f.write(f"  {g}: {c}/{t} ({100*c/t:.0f}%)\n")
        f.write(f"\noverall: {overall}/{len(y_true)} ({100*overall/len(y_true):.0f}%)\n\n")
        f.write("confusion matrix\n")
        f.write(f"{header} (predicted)\n")
        f.write("classification report\n")
        f.write(classification_report(y_true, y_pred, labels=labels_present, zero_division=0))
    print(f"\nresults saved to {out_path}")

# args
def parse_args():
    parser = argparse.ArgumentParser(description="image2gesture pipeline")
    
    parser.add_argument("--live",    action="store_true", help="run live webcam mode")
    parser.add_argument("--rebuild", action="store_true", help="rebuild Hu templates from samples")
    parser.add_argument("--image",   type=str, default=None, help="run on a single image path")
    parser.add_argument("--calibrate", action="store_true", help="launch live HSV calibration")
    parser.add_argument("--eval", action="store_true", help="run accuracy eval on data/eval/")
    
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
        sys.exit(0)

    if args.calibrate:
        run_calibrate()
        sys.exit(0)

    if args.eval:
        run_eval(templates)
        sys.exit(0)

    else:
        run_static(templates, image_path=args.image)