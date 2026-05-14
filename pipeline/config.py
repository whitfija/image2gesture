# config.py
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(BASE_DIR, "data", "samples")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
DATA_DIR = os.path.join(BASE_DIR, "data")

# HSV skin segmentation ranges
# use hsv_tuner.py to find optimal values
# i2g with--calibrate option to sample HSV values from webcam
HSV_LOWER = (0, 43, 116)
HSV_UPPER = (179, 144, 255)

# morphology
# pixel kernel sizes
MORPH_KERNEL_SIZE       = (5, 5)    # opening, keep small
MORPH_CLOSE_KERNEL_SIZE = (14, 14)  # closing, hole filling, around15

# contours
# filters small contours (probably noise)
MIN_CONTOUR_AREA = 4000
# defect depth threshold for counting fingers
# fraction of bounding box
DEFECT_MIN_SCALE = 0.9
# smoothing, fraction of contour arc length, higher = smoother
CONTOUR_SMOOTH_EPS  = 0.005

# classification
GESTURE_LABELS = ["open", "closed", "peace", "thumb", "L"]

# debug flag
SHOW_DEBUG_VIEW = True

# ROI / region of interest
# fraction of frame to ignore around edges (0.0 = no crop, 0.15 = 15% trimmed per side)
ROI_TOP    = 0.05
ROI_BOTTOM = 0.05
ROI_LEFT   = 0.05
ROI_RIGHT  = 0.05