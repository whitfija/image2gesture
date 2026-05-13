""" 
adjust sliders to find optimal HSV ranges for skin segmentation 
consider your lighting conditions and skin tone
press 's' to print values to copy into config.py, 'q' to quit

usage:
python hsv_tuner.py path/to/image.jpg
or just
python hsv_tuner.py to use first open palm sample
"""

import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import HSV_LOWER, HSV_UPPER, SAMPLES_DIR

def nothing(x):
    pass

def run_tuner(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return

    img = cv2.resize(img, (640, 480))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    cv2.namedWindow("HSV Tuner")
    cv2.createTrackbar("H_min", "HSV Tuner", HSV_LOWER[0], 179, nothing)
    cv2.createTrackbar("H_max", "HSV Tuner", HSV_UPPER[0], 179, nothing)
    cv2.createTrackbar("S_min", "HSV Tuner", HSV_LOWER[1], 255, nothing)
    cv2.createTrackbar("S_max", "HSV Tuner", HSV_UPPER[1], 255, nothing)
    cv2.createTrackbar("V_min", "HSV Tuner", HSV_LOWER[2], 255, nothing)
    cv2.createTrackbar("V_max", "HSV Tuner", HSV_UPPER[2], 255, nothing)

    print("Adjust sliders to isolate skin. Press 's' to save values, 'q' to quit.")

    while True:
        h1 = cv2.getTrackbarPos("H_min", "HSV Tuner")
        h2 = cv2.getTrackbarPos("H_max", "HSV Tuner")
        s1 = cv2.getTrackbarPos("S_min", "HSV Tuner")
        s2 = cv2.getTrackbarPos("S_max", "HSV Tuner")
        v1 = cv2.getTrackbarPos("V_min", "HSV Tuner")
        v2 = cv2.getTrackbarPos("V_max", "HSV Tuner")

        lower = np.array([h1, s1, v1])
        upper = np.array([h2, s2, v2])
        mask = cv2.inRange(hsv, lower, upper)

        # original + mask side by side
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([img, mask_bgr])
        cv2.imshow("HSV Tuner", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            print(f"\n--- copy these into config.py ---")
            print(f"HSV_LOWER = ({h1}, {s1}, {v1})")
            print(f"HSV_UPPER = ({h2}, {s2}, {v2})")
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = os.path.join(SAMPLES_DIR, "open", os.listdir(os.path.join(SAMPLES_DIR, "open"))[0])
    
    run_tuner(path)