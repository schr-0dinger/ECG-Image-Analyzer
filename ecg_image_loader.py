"""
ECG image loading and preprocessing.

Pipeline
--------
1. Load in colour.
2. Suppress ECG grid colour via HSV-channel masking (pink/red and light-blue grids).
3. Geometric correction:
     a. If a clear quadrilateral paper boundary is found → perspective warp.
     b. Otherwise → deskew rotation.
4. Median-blur (kernel 3) for impulse-noise reduction.
5. Binarize:
     a. Otsu's global threshold (fast, works on clean scans).
     b. Adaptive Gaussian threshold fallback when Otsu result is degenerate
        (< 2 % or > 60 % white pixels — common in phone-camera shots with
        uneven lighting).

Convention: dark waveform → 0, bright background → 255 (THRESH_BINARY).
"""
import logging
import math
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from deskew import determine_skew

__all__ = ['load_and_preprocess_image']

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def rotate(
    image: np.ndarray,
    angle: float,
    background: Union[int, Tuple[int, int, int]],
) -> np.ndarray:
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width  = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width)  + abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width  - old_width)  / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(
        image, rot_mat,
        (int(round(height)), int(round(width))),
        borderValue=background,
    )


def _suppress_grid_color(color_bgr: np.ndarray) -> np.ndarray:
    """
    Neutralise typical ECG grid colours so they merge with the white background.

    Standard ECG paper has a pink/red grid or occasionally a light-blue grid.
    In OpenCV HSV (H ∈ [0,180], S/V ∈ [0,255]):
      - Pink/red  : H ∈ [0,12] ∪ [168,180], S ∈ [20,150], V ≥ 120
      - Light-blue: H ∈ [90,120],            S ∈ [20,150], V ≥ 120

    Pixels matching either range are set to 255 (white) in the output
    grayscale, leaving the dark waveform trace untouched.
    """
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    red_mask  = ((h <= 12) | (h >= 168)) & (s >= 20) & (s <= 150) & (v >= 120)
    blue_mask = (h >= 90) & (h <= 120)   & (s >= 20) & (s <= 150) & (v >= 120)
    grid_mask = (red_mask | blue_mask).astype(np.uint8) * 255

    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    gray[grid_mask > 0] = 255
    return gray


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order four 2-D points as [TL, TR, BR, BL]."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    rect[0] = pts[np.argmin(s)]     # TL: min  x+y
    rect[2] = pts[np.argmax(s)]     # BR: max  x+y
    rect[1] = pts[np.argmin(diff)]  # TR: min  x-y
    rect[3] = pts[np.argmax(diff)]  # BL: max  x-y
    return rect


def _find_paper_boundary(gray: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect the four corners of the ECG paper sheet in the image.

    Works by finding the largest quadrilateral contour that covers between
    25 % and 95 % of the image area.  Returns a (4, 2) float32 array
    [TL, TR, BR, BL], or None when:
      - no clear quad is found  (→ fall back to deskew), or
      - the paper fills the frame (> 95 % area  → no correction needed).
    """
    h, w = gray.shape
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, 30, 100)
    edges   = cv2.dilate(edges, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        area = cv2.contourArea(contour)
        if area < 0.25 * h * w or area > 0.95 * h * w:
            continue
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx  = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            return _order_corners(approx.reshape(4, 2).astype(np.float32))

    return None


def _perspective_correct(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Warp the detected ECG paper region into a frontal rectangle."""
    tl, tr, br, bl = corners
    width  = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(img, M, (width, height))


def _binarize(blurred: np.ndarray) -> np.ndarray:
    """
    Binarize a grayscale image for ECG processing.

    Otsu's global threshold is preferred (fast, reliable on clean scans).
    When the result is degenerate — fewer than 2 % or more than 60 % white
    pixels — the image has strongly uneven lighting; adaptive Gaussian
    thresholding is used instead.

    Returns an image with dark waveform = 0, bright background = 255.
    """
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_ratio = np.count_nonzero(otsu) / otsu.size
    if 0.02 <= white_ratio <= 0.60:
        return otsu

    logger.debug(
        "Otsu degenerate (%.1f %% white) — switching to adaptive threshold",
        white_ratio * 100,
    )
    # block_size must be odd and smaller than the shortest image dimension
    block_size = min(151, (min(blurred.shape) // 2) * 2 - 1)
    block_size = max(block_size, 3)
    if block_size % 2 == 0:
        block_size += 1
    return cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        C=15,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_and_preprocess_image(
    image_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load, geometrically correct, and binarize an ECG image.

    Returns
    -------
    gray   : grayscale image after geometric correction.
    thresh : binarized result (dark waveform = 0, bright background = 255).
    blurred: median-blurred grayscale (stored for diagnostic use).
    """
    color = cv2.imread(image_path)
    if color is None:
        raise ValueError(f"Could not load image: {image_path}")

    # 1+2: colour load → grayscale with grid-colour suppressed to white
    gray = _suppress_grid_color(color)

    # 3: geometric correction — perspective warp preferred, deskew as fallback
    corners = _find_paper_boundary(gray)
    if corners is not None:
        logger.debug("Paper boundary detected — applying perspective correction")
        gray = _perspective_correct(gray, corners)
    else:
        angle = determine_skew(gray)
        if angle is None:
            angle = 0.0
        if abs(angle) > 0.1:
            logger.debug("Deskew angle: %.2f°", angle)
        # White border fill avoids black edge artefacts in grid detection
        gray = rotate(gray, angle, (255, 255, 255))

    # 4: median blur — kernel 3 removes impulse noise without blurring edges
    blurred = cv2.medianBlur(gray, 3)

    # 5: binarize
    thresh = _binarize(blurred)

    return gray, thresh, blurred
