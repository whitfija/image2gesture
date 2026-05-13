# image2gesture

A multi-stage classical image processing pipeline for real-time hand gesture recognition using a standard webcam. Recognizes 5 hand gestures and applies live image filter effects driven by the detected gesture.

*For I2200/EE4710: Introduction to Digital Image Processing — City College of New York*
*Spring 2026*

---

## gestures

```
Gesture     -   Filter Effect
-----------------------------------------
Open palm   -   Gaussian blur
Closed fist -   Histogram equalization
Peace sign  -   Ordered dithering (Bayer)
Thumbs up   -   Negative
L           -   Halftone dots
```

---

## pipeline

```
image frame --> HSV segmentation --> morphological cleanup --> contour extraction --> classification --> filter output
```

- **Segmentation** - Gaussian blur pre-pass, HSV color segmentation with per-session calibration, ROI masking
- **Morphology** - morphological opening (speckle removal) and closing (hole filling)
- **Contour extraction** - convex hull, convexity defects, finger counting, bounding box aspect ratio, Hu moments
- **Classification** - rule-based primary classifier (finger count + aspect ratio); Hu moment tiebreaker for ambiguous gesture pairs
- **Filter output** - live filter view driven by gesture label, togglable alongside debug view

---

## usage

```bash
python i2g.py                        # runs test images in data/input/
python i2g.py --image path/to/img    # single image
python i2g.py --live                 # live webcam loop
python i2g.py --rebuild              # rebuild Hu moment templates from samples
python i2g.py --calibrate            # live HSV calibration with sliders
python i2g.py --eval                 # accuracy eval against data/eval/ ground truth folders
```

### live mode controls

| Key   | Action                                  |
| ----- | --------------------------------------- |
| `d` | Debug view (segmentation pipeline grid) |
| `f` | Filter view (gesture-driven effect)     |
| `b` | Both views side by side                 |
| `s` | Save current frame(s) to 'output/'      |
| `q` | Quit                                    |

### calibrate mode controls

| Key   | Action                                 |
| ----- | -------------------------------------- |
| `p` | Pause / unpause feed                   |
| `s` | Save HSV values to `hsv_config.json` |
| `q` | Quit                                   |

> HSV calibration should be run at the start of each session, skin segmentation is lighting-sensitive.

---

## eval mode

Expects ground-truth labeled folders under `data/eval/`:

```
data/eval/
├── open/
├── closed/
├── peace/
├── thumb/
└── L/
```

Outputs per-gesture accuracy, a confusion matrix, and a full classification report. Results saved to `output/eval_results.txt`.

---

## dependencies

install via:

```bash
pip install -r requirements.txt
```

---

## config

All primary tuning parameters live in `config.py`. Classification thresholds are in `classification.py`.

### HSV segmentation

```python
HSV_LOWER = (0, 43, 116)
HSV_UPPER = (179, 144, 255)
```

HSV range for skin segmentation. Actual values are loaded from `hsv_config.json` at runtime

Run `--calibrate` to tune for your lighting and skin tone and save a session config. Uses `config.py` values as backup

### ROI

```python
ROI_TOP    = 0.05
ROI_BOTTOM = 0.05
ROI_LEFT   = 0.05
ROI_RIGHT  = 0.05
```

Fraction of the frame to ignore around each edge

### morphology

```python
MORPH_KERNEL_SIZE       = (5, 5)
MORPH_CLOSE_KERNEL_SIZE = (14, 14)
```

`MORPH_KERNEL_SIZE` controls the opening pass (erode to dilate), which removes small noise blobs. Keep this small or it will erode real hand features

`MORPH_CLOSE_KERNEL_SIZE` controls the closing pass (dilate to erode), which fills gaps in the hand mask

### contours

```python
MIN_CONTOUR_AREA   = 4000
DEFECT_MIN_SCALE   = 0.7
CONTOUR_SMOOTH_EPS = 0.005
```

`MIN_CONTOUR_AREA` - contours smaller than this (in pixels) are rejected as noise

`DEFECT_MIN_SCALE` — convexity defects shallower than `contour_area * DEFECT_MIN_SCALE` are ignored when counting fingers

`CONTOUR_SMOOTH_EPS` — smoothing applied via `approxPolyDP`, fraction of contour arc length

### classification thresholds

```python
FINGER_COUNT_MAP = {
    "open":   (3, 5),
    "peace":  (1, 3),
    "thumb":  (0, 2),
    "L":      (1, 3),
    "closed": (0, 1),
}

ASPECT_RATIO_MAP = {
    "open":   (0.6, 1.0),
    "peace":  (0.2, 0.6),
    "thumb":  (0.25, 0.38),
    "L":      (0.55, 1.0),
    "closed": (0.6, 1.1),
}
```

`FINGER_COUNT_MAP` - acceptable finger count range (min, max) per gesture
`ASPECT_RATIO_MAP` - acceptable width/height bounding box ratio per gesture.
*both used for rule-based classification*
