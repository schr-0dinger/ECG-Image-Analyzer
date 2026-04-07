import logging

import cv2
import numpy as np

__all__ = ['isolate_waveform']

logger = logging.getLogger(__name__)


def isolate_waveform(image, dx=None, dy=None, debug=True):
    """
    Suppress the ECG grid and enhance the waveform. Assumes input is grayscale or binary.
    Returns a binary image where the ECG trace is white on black.

    Args:
        image: grayscale or binary uint8 image.
        dx: pixels per 1 mm horizontally (from grid calibration). Used to size kernels.
        dy: pixels per 1 mm vertically (from grid calibration). Used to size kernels.
        debug: if True, display intermediate images via matplotlib.
    """
    # If image is not already binary, threshold it
    if image.dtype != np.uint8 or np.max(image) > 1:
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    else:
        binary = image.copy()

    # Kernel size = ~3 mm in each direction so that continuous grid lines are detected
    # but the curved waveform (which only runs straight for <2 mm before turning) is not.
    # Falls back to 33 px if calibration is unavailable.
    if dx is not None and dy is not None:
        kw = max(3, int(round(dx * 3)))
        kh = max(3, int(round(dy * 3)))
        logger.debug("Adaptive kernels: horizontal=%dpx, vertical=%dpx", kw, kh)
    else:
        kw = kh = 33
        logger.warning("dx/dy not provided — using fallback kernel size of 33 px")

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kh))
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1)

    grid_mask = cv2.bitwise_or(horizontal_lines, vertical_lines)
    waveform  = cv2.bitwise_and(binary, cv2.bitwise_not(grid_mask))

    # Phase 1.3: Remove noise specks via connected-component size filtering.
    # Minimum component area = 5 % of a 1 mm² cell (in pixels²).
    # This discards isolated dots from paper texture, ink splatter, and JPEG
    # artefacts while keeping all genuine waveform segments.
    if dx is not None and dy is not None:
        min_area = max(3, int(dx * dy * 0.05))
    else:
        min_area = 3

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(waveform, connectivity=8)
    cleaned = np.zeros_like(waveform)
    kept = 0
    for lbl in range(1, num_labels):           # label 0 is the background
        if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == lbl] = 255
            kept += 1
    waveform = cleaned
    logger.debug(
        "Noise filter: kept %d / %d components (min_area=%d px²)",
        kept, num_labels - 1, min_area,
    )

    if debug:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,4))
        plt.subplot(1,4,1); plt.title('Binary'); plt.imshow(binary, cmap='gray')
        plt.subplot(1,4,2); plt.title('Horizontal Grid'); plt.imshow(horizontal_lines, cmap='gray')
        plt.subplot(1,4,3); plt.title('Vertical Grid'); plt.imshow(vertical_lines, cmap='gray')
        plt.subplot(1,4,4); plt.title('Waveform'); plt.imshow(waveform, cmap='gray')
        plt.tight_layout()
        plt.show()

    return waveform