import logging

import numpy as np
import cv2
from scipy.ndimage import median_filter

__all__ = ['extract_ecg_signal', 'estimate_baseline']

logger = logging.getLogger(__name__)


def estimate_baseline(waveform_image):
    """
    Estimate the baseline (isoelectric line) as the row with the highest pixel density.
    """
    projection = np.sum(waveform_image > 0, axis=1)
    baseline_y = np.argmax(projection)
    return baseline_y


def _subpixel_refine(x, grayscale_image):
    """
    Refine y-coordinate using parabolic fitting on inverted grayscale intensity.
    Uses the intensity profile at column x to estimate sub-pixel position.
    """
    h, w = grayscale_image.shape
    if x < 1 or x >= w - 1:
        return None

    intensities = []
    positions = []
    for dx in [-1, 0, 1]:
        col = grayscale_image[:, x + dx]
        if np.max(col) > 0:
            center_of_mass = np.sum(np.arange(h) * col) / (np.sum(col) + 1e-9)
            intensities.append(np.sum(col))
            positions.append(center_of_mass)

    if len(intensities) >= 2 and np.max(intensities) > 0:
        max_idx = np.argmax(intensities)
        y_estimate = positions[max_idx]
        return y_estimate

    return None


def _extract_cluster_median(ys, prev_y):
    """
    Handle multi-pixel columns by detecting clusters separated by gaps > 2 pixels.
    Select the cluster whose median is closest to previous column's value for continuity.
    """
    if len(ys) == 0:
        return prev_y

    ys_sorted = np.sort(ys)
    gaps = np.where(np.diff(ys_sorted) > 2)[0] + 1
    clusters = np.split(ys_sorted, gaps)

    if len(clusters) == 1:
        return np.median(clusters[0])

    best_cluster = None
    best_dist = np.inf
    for cluster in clusters:
        cluster_median = np.median(cluster)
        dist = abs(cluster_median - prev_y)
        if dist < best_dist:
            best_dist = dist
            best_cluster = cluster_median

    return best_cluster if best_cluster is not None else np.median(ys_sorted)


def extract_ecg_signal(waveform_image, dx, dy, grayscale_image=None, debug=False):
    """
    Extract a 1D ECG signal (in mV vs time) from a grid-removed binary waveform image.

    Args:
        waveform_image (np.ndarray): Binary image (waveform = white on black) after grid removal.
        dx (float): pixels per 1 mm in horizontal direction.
        dy (float): pixels per 1 mm in vertical direction.
        grayscale_image (np.ndarray): Optional grayscale image for sub-pixel interpolation.
        debug (bool): If True, show diagnostic plots.

    Returns:
        times (np.ndarray): time axis in seconds.
        volts (np.ndarray): voltage axis in mV.
        baseline_y (int): pixel row of estimated baseline.
    """
    sec_per_mm = 0.04
    mv_per_mm = 0.1

    s_per_px = sec_per_mm / dx
    mv_per_px = mv_per_mm / dy

    h, w = waveform_image.shape

    baseline_y = estimate_baseline(waveform_image)

    y_coords = np.full(w, baseline_y, dtype=float)
    prev_y = baseline_y
    for x in range(w):
        ys = np.where(waveform_image[:, x] > 0)[0]
        if ys.size:
            y_coords[x] = _extract_cluster_median(ys, prev_y)
            prev_y = y_coords[x]

    if grayscale_image is not None:
        for x in range(1, w - 1):
            if y_coords[x] != baseline_y:
                refined = _subpixel_refine(x, grayscale_image)
                if refined is not None:
                    y_coords[x] = refined

    volts = (baseline_y - y_coords) * mv_per_px

    window_size = max(3, int(2 * dx))
    baseline_drift = median_filter(volts, size=window_size)
    volts_corrected = volts - baseline_drift

    times = np.arange(w) * s_per_px

    if debug:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 1, 1)
        plt.plot(times, volts, label='Original')
        plt.plot(times, baseline_drift, label='Baseline drift')
        plt.title("Extracted ECG Signal - Before baseline correction")
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (mV)")
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(times, volts_corrected, label='Baseline corrected')
        plt.title("Extracted ECG Signal - After baseline correction")
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (mV)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return times, volts_corrected, baseline_y