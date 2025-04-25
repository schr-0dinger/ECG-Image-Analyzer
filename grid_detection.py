import numpy as np
import cv2
from scipy.signal import find_peaks

def get_grid_spacing(thresh, debug=False):
    """
    thresh: binary image (waveform+grid = white on black).
    returns: (dx, dy) pixels per 1 mm small box, forced square if distortion >
    """

    # 1) enhance grid → suppress trace, close gridlines
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1,3))
    no_wave = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_v)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (3,1))
    grid = cv2.morphologyEx(no_wave, cv2.MORPH_CLOSE, kernel_h)

    # 2) projections on grid only
    proj_x = grid.sum(axis=0)
    proj_y = grid.sum(axis=1)

    def spacing_from_projection(proj, label):
        # smooth
        proj = np.convolve(proj, np.ones(5)/5, mode='same')
        fft = np.abs(np.fft.fft(proj))
        fft[0] = 0
        freqs = np.fft.fftfreq(len(proj))
        pos = slice(1, len(freqs)//2)
        freqs, fft = freqs[pos], fft[pos]

        # limit to reasonable mm range
        valid = np.where((freqs>=1/50)&(freqs<=1/5))[0]
        if valid.size==0:
            return 0
        peaks, props = find_peaks(fft[valid], prominence=fft[valid].max()*0.1)
        if not len(peaks):
            return 0

        # choose the most prominent
        peak = valid[peaks[np.argmax(props["prominences"])]]
        return 1/abs(freqs[peak])

    dx = spacing_from_projection(proj_y, "horiz")
    dy = spacing_from_projection(proj_x, "vert")

    # 3) sanity-check squareness
    if dx and dy:
        if max(dx,dy)/min(dx,dy) > 1.1:
            # too distorted → fallback to average
            avg = (dx+dy)/2
            if debug:
                print(f"⚠️ distortion detected, forcing square: {dx:.1f},{dy:.1f} → {avg:.1f}")
            dx = dy = avg

    return dx, dy

def crop_strips(binary, n_strips=5, overlap=0.2):
    """
    Return list of n_strips horizontal sub-images of `binary`.  
    overlap is fraction of strip height to overlap (to avoid missing grid lines at edges).
    """
    h, w = binary.shape # w not used
    if n_strips < 1:
        raise ValueError("n_strips must be at least 1")
    strip_h = int(np.ceil(h / n_strips))
    step = int(strip_h * (1 - overlap))
    strips = []
    for y in range(0, h - strip_h + 1, step):
        strips.append(binary[y : y + strip_h, :])
    return strips

def robust_grid_spacing(binary, debug=False):
    # 1) Crop into strips
    strips = crop_strips(binary, n_strips=9, overlap=0.3)
    
    # 2) Compute dx, dy per strip
    results = []
    for i, strip in enumerate(strips):
        dx, dy = get_grid_spacing(strip, debug=debug)
        if dx>0 and dy>0:
            results.append((dx, dy))
        elif debug:
            print(f"  strip {i}: invalid spacing dx={dx},dy={dy}")
    
    if not results:
        raise RuntimeError("No valid grid spacing detected in any strip.")
    
    # 3) Separate dxs and dys & median-filter outliers
    dxs = np.array([r[0] for r in results])
    dys = np.array([r[1] for r in results])
    med_dx = np.median(dxs)
    med_dy = np.median(dys)
    # keep only those within ±10% of median
    good = (np.abs(dxs - med_dx) / med_dx < 0.1) & (np.abs(dys - med_dy) / med_dy < 0.1)
    dxs, dys = dxs[good], dys[good]
    if len(dxs)==0:
        raise RuntimeError("All strip measurements were outliers.")
    
    # 4) Final dx, dy = median of the filtered set
    final_dx = float(np.median(dxs))
    final_dy = float(np.median(dys))
    
    if debug:
        print("Per-strip dxs:", dxs)
        print("Per-strip dys:", dys)
        print(f"→ Final grid spacing: dx={final_dx:.2f}, dy={final_dy:.2f}")
    
    return final_dx, final_dy