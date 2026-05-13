# config.py
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(BASE_DIR, "data", "samples")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
DATA_DIR = os.path.join(BASE_DIR, "data")

# HSV skin segmentation ranges
# use hsv_tuner.py to find optimal values
HSV_LOWER = (0, 43, 116)
HSV_UPPER = (179, 144, 255)

# morphology
MORPH_KERNEL_SIZE = (5, 5)

# contours
# filters small countours (probably noise)
MIN_CONTOUR_AREA = 2000

# classification
# gestures
GESTURE_LABELS = ["open", "closed", "peace", "thumb", "L"]
# defect depth threshold for counting fingers
DEFECT_MIN_DEPTH = 10000

# debug flag
SHOW_DEBUG_VIEW = True

# ROI / region of interest
# fraction of frame to ignore around edges (0.0 = no crop, 0.15 = 15% trimmed per side)
ROI_TOP    = 0.05
ROI_BOTTOM = 0.05
ROI_LEFT   = 0.05
ROI_RIGHT  = 0.05