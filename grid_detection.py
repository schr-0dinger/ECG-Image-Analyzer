import logging

import numpy as np
import cv2
from scipy.signal import find_peaks

__all__ = ['robust_grid_spacing', 'get_grid_spacing']

logger = logging.getLogger(__name__)


def get_grid_spacing(thresh, debug=False):
    """
    Detect ECG grid spacing from a binary image via FFT-based periodicity analysis.

    Args:
        thresh: binary image (waveform + grid structures visible).
        debug:  emit debug log messages when True.

    Returns:
        dx         : pixels per 1 mm horizontally (0 if detection failed).
        dy         : pixels per 1 mm vertically   (0 if detection failed).
        confidence : float in [0, 1] — how strongly periodic the detected
                     spacing is.  Values below ~0.4 indicate unreliable results.
    """
    # 1) Enhance grid — suppress the waveform trace, connect broken grid lines
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    no_wave  = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_v)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    grid     = cv2.morphologyEx(no_wave, cv2.MORPH_CLOSE, kernel_h)

    # 2) Row / column projections on the grid-only image
    proj_x = grid.sum(axis=0)
    proj_y = grid.sum(axis=1)

    def spacing_from_projection(proj):
        """
        Return (spacing_px, confidence) from a 1-D projection array.

        confidence is the normalised prominence of the dominant FFT peak
        multiplied by a harmonic-validation factor (0.7 when no harmonic is
        found, up to 1.0 when a clear first harmonic is present).
        """
        proj = np.convolve(proj, np.ones(5) / 5, mode='same')
        fft  = np.abs(np.fft.fft(proj))
        fft[0] = 0
        freqs = np.fft.fftfreq(len(proj))

        # Restrict to the physically meaningful band: 5–50 px/cycle (1–10 mm at any DPI)
        pos   = slice(1, len(freqs) // 2)
        freqs = freqs[pos]
        fft   = fft[pos]
        valid = np.where((freqs >= 1 / 50) & (freqs <= 1 / 5))[0]
        if valid.size == 0:
            return 0, 0.0

        peaks, props = find_peaks(fft[valid], prominence=fft[valid].max() * 0.1)
        if not len(peaks):
            return 0, 0.0

        best      = valid[peaks[np.argmax(props["prominences"])]]
        spacing   = 1.0 / abs(freqs[best])

        # Base confidence: normalised peak prominence in the valid band
        peak_prom = props["prominences"][np.argmax(props["prominences"])]
        base_conf = min(1.0, peak_prom / (fft[valid].max() + 1e-9))

        # Harmonic validation: first harmonic at 2× the fundamental frequency
        fund_freq      = abs(freqs[best])
        harmonic_freq  = 2.0 * fund_freq
        h_range = np.where(
            (freqs >= harmonic_freq * 0.85) & (freqs <= harmonic_freq * 1.15)
        )[0]
        if len(h_range) > 0:
            h_power        = fft[h_range].max()
            harmonic_ratio = h_power / (fft[best] + 1e-9)
            # A harmonic at ≥ 10 % of the fundamental → fully confident
            harmonic_factor = 0.7 + 0.3 * min(1.0, harmonic_ratio / 0.1)
        else:
            harmonic_factor = 0.7   # penalise absent harmonic

        return spacing, base_conf * harmonic_factor

    dx, dx_conf = spacing_from_projection(proj_y)
    dy, dy_conf = spacing_from_projection(proj_x)

    # 3) Sanity-check squareness: ECG paper is physically square-gridded
    if dx and dy:
        if max(dx, dy) / min(dx, dy) > 1.1:
            avg = (dx + dy) / 2
            logger.debug("Distortion detected, forcing square: %.1f, %.1f → %.1f", dx, dy, avg)
            dx = dy = avg

    confidence = (dx_conf + dy_conf) / 2.0
    return dx, dy, confidence


def crop_strips(binary, n_strips=5, overlap=0.2):
    """
    Slice a binary image into *n_strips* horizontal sub-images with
    fractional *overlap* between adjacent strips.
    """
    h, _ = binary.shape
    if n_strips < 1:
        raise ValueError("n_strips must be at least 1")
    strip_h = int(np.ceil(h / n_strips))
    step    = int(strip_h * (1 - overlap))
    return [binary[y: y + strip_h, :] for y in range(0, h - strip_h + 1, step)]


def robust_grid_spacing(binary, debug=False):
    """
    Estimate ECG grid spacing robustly by processing multiple horizontal strips
    and taking the median of inlier measurements.

    Args:
        binary: binary image as produced by load_and_preprocess_image.
        debug:  emit per-strip debug logs when True.

    Returns:
        dx         : pixels per 1 mm horizontally.
        dy         : pixels per 1 mm vertically.
        confidence : median confidence score across accepted strips (0–1).
                     Values below 0.4 indicate the grid may not be reliably
                     detected; callers should warn the user.

    Raises:
        RuntimeError if no valid spacing can be found.
    """
    strips  = crop_strips(binary, n_strips=9, overlap=0.3)
    results = []

    for i, strip in enumerate(strips):
        dx, dy, conf = get_grid_spacing(strip, debug=debug)
        if dx > 0 and dy > 0:
            results.append((dx, dy, conf))
        elif debug:
            logger.debug("Strip %d: invalid spacing dx=%s dy=%s", i, dx, dy)

    if not results:
        raise RuntimeError("No valid grid spacing detected in any strip.")

    dxs   = np.array([r[0] for r in results])
    dys   = np.array([r[1] for r in results])
    confs = np.array([r[2] for r in results])

    med_dx = np.median(dxs)
    med_dy = np.median(dys)
    # Keep only strips within ±10 % of the median (reject outlier strips)
    good  = (np.abs(dxs - med_dx) / med_dx < 0.1) & (np.abs(dys - med_dy) / med_dy < 0.1)
    dxs, dys, confs = dxs[good], dys[good], confs[good]

    if len(dxs) == 0:
        raise RuntimeError("All strip measurements were outliers.")

    final_dx   = float(np.median(dxs))
    final_dy   = float(np.median(dys))
    final_conf = float(np.median(confs))

    if debug:
        logger.debug("Per-strip dxs: %s", dxs)
        logger.debug("Per-strip dys: %s", dys)
        logger.debug("Final grid spacing: dx=%.2f dy=%.2f confidence=%.2f",
                     final_dx, final_dy, final_conf)

    return final_dx, final_dy, final_conf
