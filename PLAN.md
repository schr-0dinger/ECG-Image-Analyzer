# ECG Image Analyzer — Production Roadmap

> **Goal:** Transform a research prototype into a production-grade ECG interpretation tool capable of handling real-world clinical images with diagnostic-level accuracy.

---

## Phase 0: Code Triage (Fix What's Broken Now)
*These are silent failures in the current code. Nothing in Phase 1+ works correctly until these are resolved.*

### Critical Bugs

- **Duplicate `detect_r_peaks()` in `main.py` (lines 77 & 173):** Python silently discards the Pan-Tompkins implementation (lines 77–107) in favor of the simpler `find_peaks` version below it. Decision: unify into a single method, expose algorithm choice via a parameter.

- **`run_pipeline()` is broken (line 214):** It calls `interpolate_waveform()` before `isolate_and_extract_waveform()`, so `self.waveform` is always `None`. The call to `isolate_and_extract_waveform()` is missing entirely. Fix call order:
  ```
  load_and_prepare_image()
  → detect_grid_spacing()
  → isolate_and_extract_waveform()   ← missing
  → interpolate_waveform()
  → select_roi()
  → extract_1d_signal()
  → detect_r_peaks()
  → calculate_heart_rate()
  → plot_results()
  ```

- **`isolate_waveform.py` ignores `dx`/`dy`:** Hardcoded 33-pixel kernels break on any scan that isn't ~330 DPI. The parameters are accepted but never used — replace with `kernel_h = max(3, int(round(dx * 0.8))) | 1` style calculation.

- **`calculate_sampling_frequency()` never called by active `detect_r_peaks()`:** The second (live) version uses `self.dx * 2` as raw distance with no physical unit derivation. This gives wrong refractory periods on non-standard DPI images.

- **Replace all `print()` statements** with Python `logging` module (configurable level). Print-based status makes library integration impossible.

### Quick Wins
- Move `relative_mode()` out of `run_pipeline()` into a proper module-level utility.
- Add `__all__` exports to each module.
- Validate inputs at pipeline entry: file exists, is readable, is a valid image.

---

## Phase 1: Robust Real-World Image Handling
*Goal: Reliably ingest ECG images from phone cameras, photocopies, and low-quality scans — not just clean high-DPI scans.*

### 1.1 Advanced Preprocessing (`ecg_image_loader.py`)
- **Perspective correction:** Detect ECG paper boundary via quadrilateral contour detection (not just deskew). A phone photo taken at an angle will fail deskew-only correction. Use `cv2.getPerspectiveTransform` + `warpPerspective`.
- **Adaptive binarization:** Replace single global Otsu threshold with a combination: `cv2.adaptiveThreshold` (local block method) for uneven lighting, with Otsu as fallback. Handles poor lighting common in clinical settings.
- **Color-aware preprocessing:** ECG paper often has colored grid (red/pink). Convert via HSV channel separation to suppress the grid color before thresholding rather than relying purely on morphology later.
- **Resolution normalization:** Detect DPI via grid calibration and resample to a canonical internal resolution (e.g., 300 DPI equivalent) before all downstream processing. This decouples all subsequent steps from input DPI variation.

### 1.2 Robust Grid Detection (`grid_detection.py`)
- **Multi-scale FFT:** Current FFT assumes a single dominant frequency. Add peak harmonic validation — confirm the detected period divides evenly into the image width, rejecting spurious frequencies.
- **Fallback calibration mode:** If grid is not detectable (smudged, overexposed), allow the user to specify scale via a calibration mark (e.g., 1mV pulse or reference box drawn by the user). Expose as `GridCalibration(mode='auto' | 'manual', mm_per_px=...)`.
- **Grid confidence score:** Return a quality metric (0–1) alongside `dx`/`dy`. Downstream steps should log a warning when confidence < 0.7.

### 1.3 Noise & Artifact Handling
- **Morphological cleanup pass:** After binarization and before grid removal, add an opening pass to remove isolated noise pixels smaller than `min(dx, dy) * 0.3`.
- **Stamp/annotation masking:** Large connected components that are clearly text or stamps (wrong aspect ratio for waveform segments) should be masked out before waveform isolation.

---

## Phase 2: Accurate Signal Digitization
*Goal: Get a clean, calibrated 1D signal from each lead that matches the paper record within clinical tolerances.*

### 2.1 Adaptive Waveform Isolation (`isolate_waveform.py`)
- **Fix adaptive kernels:** Use `dx`/`dy` to compute horizontal/vertical structuring element sizes:
  ```python
  h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(dx * 0.8) | 1, 1))
  v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(dy * 0.8) | 1))
  ```
- **Zhang-Suen skeletonization:** After grid removal, thin the waveform to 1 pixel width using `skimage.morphology.thin()` (already in `requirements.txt`). This eliminates the column-median ambiguity in thick traces.
- **Broken trace repair:** After skeletonization, bridge gaps < `dx * 0.3` pixels using morphological dilation + connected component re-linking.

### 2.2 Signal Extraction Improvements (`waveform_extraction.py`)
- **Baseline wander correction:** After extracting raw volts, apply a moving-median filter with window `= 2 * fs` (one heartbeat) to estimate and subtract baseline drift. This is separate from and complementary to the bandpass filter.
- **Multi-column conflict resolution:** When multiple waveform pixels exist in a column (overlapping leads), use the pixel cluster with the largest connected component rather than a naive median.
- **Sub-pixel interpolation:** Fit a parabola to the 3 pixels around each column's peak y-coordinate for sub-pixel accuracy.

### 2.3 R-Peak Detection Unification (`main.py`)
- **Single `detect_r_peaks(algorithm='pantompkins'|'simple')` method** combining the two current implementations.
- **Pan-Tompkins corrections:** The current bandpass uses 5–15 Hz which is correct for QRS enhancement, but the adaptive threshold window (2s) is too short. Use 8s window per AHA specification.
- **Physiological HR validation:** Emit a structured warning dict, not just `print()`. Downstream code (API, web UI) needs machine-readable quality flags.
- **Signal quality index (SQI):** Before peak detection, compute a simple SQI: ratio of signal power in 5–15 Hz band to total power. Return SQI alongside peaks so callers can gate on quality.

---

## Phase 3: Clinical Feature Extraction
*Goal: Produce diagnostic-grade interval measurements and morphology analysis.*

### 3.1 Diagnostic Filtering
- **0.05–150 Hz bandpass** (AHA/ACC standard for diagnostic ECG display). The current 5–15 Hz is only appropriate for QRS detection, not for P/T wave analysis.
- Implement as a reusable `filter_ecg(signal, fs, mode='diagnostic'|'monitor'|'qrs')` function with named presets.

### 3.2 12-Lead Layout Parser (new module: `lead_parser.py`)
- Detect standard layout: 3×4 grid (I, II, III | aVR, aVL, aVF | V1–V6) + optional rhythm strip.
- Use horizontal/vertical projection profiles of the isolated waveform to locate lead boundaries.
- Return a dict `{'I': (x, y, w, h), 'II': ..., ...}` mapping lead names to bounding boxes.
- Must handle layout variants: Cabrera order, 6×2, single-lead rhythm strips.

### 3.3 Feature Extraction Engine (new module: `ecg_features.py`)
Compute the following from a calibrated, filtered 1D signal:

| Feature | Method |
|---|---|
| Heart Rate | Median RR interval; flag irregular rhythm |
| PR Interval | P-wave onset to QRS onset |
| QRS Duration | QRS onset to QRS offset via derivative threshold |
| QT Interval | QRS onset to T-wave end (tangent method) |
| QTc | Bazett (`QT/√RR`), Fridericia (`QT/∛RR`) — both |
| ST Deviation | Mean amplitude 60–80 ms post J-point vs. isoelectric |
| P/T-wave morphology | Peak amplitude, width, axis estimate |

- Each feature should carry a confidence value and a data-quality flag.
- Return a structured `ECGReport` dataclass, not raw numbers.

---

## Phase 4: Production Architecture
*Goal: Expose the engine as a reliable, deployable service.*

### 4.1 Project Structure Refactor
Reorganize into a proper package before adding more code:
```
ecg_analyzer/
├── core/
│   ├── loader.py          (from ecg_image_loader.py)
│   ├── grid.py            (from grid_detection.py)
│   ├── isolator.py        (from isolate_waveform.py)
│   ├── extractor.py       (from waveform_extraction.py)
│   ├── features.py        (new)
│   └── lead_parser.py     (new)
├── pipeline.py            (ECGProcessor, from main.py)
├── api/
│   ├── app.py             (FastAPI application)
│   └── schemas.py         (Pydantic request/response models)
├── cli.py                 (Batch processing CLI via Typer/Click)
├── web/                   (Streamlit dashboard)
└── tests/
```

### 4.2 FastAPI Backend (`api/`)
- `POST /analyze` — accepts image upload, returns `ECGReport` JSON.
- `POST /analyze/batch` — accepts a ZIP of images, returns array of reports.
- `GET /health` — liveness probe.
- Async processing with `asyncio` for non-blocking image analysis.
- Pydantic response schemas for all outputs (structured, versioned).
- OpenAPI docs auto-generated via FastAPI.

### 4.3 Streamlit Web UI (`web/`)
- Upload widget → live preview of preprocessing stages.
- Side-by-side: original image | grid overlay | isolated waveform.
- Interactive signal plot with annotated peaks and intervals.
- Summary table of computed features with normal-range highlighting.
- Export button: JSON / CSV / PDF report.

### 4.4 CLI for Batch Processing (`cli.py`)
```bash
ecg-analyze image.png                         # single image
ecg-analyze ./records/ --format json          # folder, JSON output
ecg-analyze ./records/ --format csv --leads II V1  # filtered leads
```
Use `typer` for argument parsing. Stream results to stdout (JSONL) for pipeline composability.

### 4.5 Standardized Export
- **JSON:** Full `ECGReport` with all features, quality flags, and metadata.
- **CSV:** Tabular time/voltage pairs per lead, one file per lead.
- **WFDB:** Write `.dat` + `.hea` files compatible with PhysioNet tools via `wfdb` library.

### 4.6 Containerization
- `Dockerfile`: multi-stage build — builder stage installs deps, runtime stage is minimal.
- `docker-compose.yml`: API + web UI services.
- No GUI dependencies (`cv2.imshow`, `plt.show()`, `ROISelector`) in container; these must be behind `if not headless:` guards.

---

## Phase 5: Validation & Reliability
*Goal: Quantify accuracy, prevent regressions, build trust.*

### 5.1 Test Suite (`tests/`)
- **Unit tests:** Each module (`grid.py`, `isolator.py`, `extractor.py`, `features.py`) tested in isolation with synthetic inputs.
- **Integration tests:** Full pipeline on known reference images; assert `heart_rate ± 5 bpm`, `QRS_duration ± 10 ms`.
- **Regression images:** Commit a small set of reference ECG PNGs + expected output JSONs. CI fails if outputs drift.

### 5.2 Benchmark Against Reference Datasets
- **PhysioNet PTB-XL:** 21,000+ 12-lead ECG records with known metadata. Evaluate HR, PR, QRS, QTc accuracy.
- **MIT-BIH Arrhythmia DB:** Use waveform reconstruction to compare digitized signal against reference — report RMSE and SNR.
- Publish a `BENCHMARK.md` with methodology and results.

### 5.3 CI/CD (GitHub Actions)
```yaml
# .github/workflows/ci.yml
on: [push, pull_request]
jobs:
  test:  pytest tests/ --cov=ecg_analyzer
  lint:  ruff check . && mypy ecg_analyzer/
  build: docker build .
```

### 5.4 Documentation
- **API docs:** Auto-generated via FastAPI + Sphinx for the Python package.
- **Accuracy statement:** Published in README — what the tool can and cannot diagnose. Explicit disclaimer: not a medical device, for research use.
- **Contribution guide:** Setup instructions, module overview, how to add a new feature extractor.

---

## Implementation Order (Recommended)

```
Phase 0  →  Phase 1.1 + 1.2  →  Phase 2.1 + 2.2  →  Phase 2.3
   ↓
Phase 3.1 + 3.2  →  Phase 3.3
   ↓
Phase 4.1 (refactor structure)  →  Phase 4.2 + 4.3 + 4.4 + 4.5
   ↓
Phase 4.6  →  Phase 5.1 + 5.3  →  Phase 5.2 + 5.4
```

> Phase 0 is a prerequisite for everything. Phase 4.1 (restructure) should happen before the API/UI work to avoid large merge conflicts.
