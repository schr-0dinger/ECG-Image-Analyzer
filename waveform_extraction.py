import numpy as np
import cv2

def estimate_baseline(waveform_image):
    """
    Estimate the baseline (isoelectric line) as the row with the highest pixel density.
    """
    # Count white pixels in each row
    projection = np.sum(waveform_image > 0, axis=1)
    baseline_y = np.argmax(projection)
    return baseline_y


def extract_ecg_signal(waveform_image, dx, dy, debug=False):
    """
    Extract a 1D ECG signal (in mV vs time) from a grid-removed binary waveform image.

    Args:
        waveform_image (np.ndarray): Binary image (waveform = white on black) after grid removal.
        dx (float): pixels per 1 mm in horizontal direction.
        dy (float): pixels per 1 mm in vertical direction.
        debug (bool): If True, show diagnostic plots.

    Returns:
        times (np.ndarray): time axis in seconds.
        volts (np.ndarray): voltage axis in mV.
        baseline_y (int): pixel row of estimated baseline.
    """
    # Constants: paper speed 25 mm/s, sensitivity 10 mm/mV
    sec_per_mm = 0.04          # 1 mm = 0.04 s
    mv_per_mm = 0.1            # 1 mm = 0.1 mV

    # Convert pixel scales to real units
    s_per_px = sec_per_mm / dx   # seconds per pixel horizontally
    mv_per_px = mv_per_mm / dy   # mV per pixel vertically

    h, w = waveform_image.shape

    # Estimate baseline
    baseline_y = estimate_baseline(waveform_image)

    # Extract signal: for each column, find waveform pixels and compute median y
    y_coords = np.full(w, baseline_y, dtype=float)
    for x in range(w):
        ys = np.where(waveform_image[:, x] > 0)[0]
        if ys.size:
            # choose the median pixel to reduce noise
            y_coords[x] = np.median(ys)

    # Compute voltage relative to baseline (positive upward)
    # Note: image origin (0,0) is top-left, so downward y increases.  baseline_y - y gives deflection sign.
    volts = (baseline_y - y_coords) * mv_per_px

    # Time axis
    times = np.arange(w) * s_per_px

    if debug:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        plt.plot(times, volts)
        plt.title("Extracted ECG Signal")
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (mV)")
        plt.grid(True)
        plt.show()

    return times, volts, baseline_y
