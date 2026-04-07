import logging
import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt

from ecg_image_loader import load_and_preprocess_image
from grid_detection import robust_grid_spacing
from isolate_waveform import isolate_waveform
from waveform_extraction import extract_ecg_signal
from roi_selector import ROISelector

logger = logging.getLogger(__name__)


def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


def relative_mode(heart_rates, threshold=5):
    """Return the mean of the largest cluster of similar heart rates."""
    groups = []
    for hr in heart_rates:
        group = [x for x in heart_rates if abs(x - hr) <= threshold]
        if len(group) > 1:
            groups.append(group)
    if groups:
        return np.mean(max(groups, key=len))
    return np.mean(heart_rates)


class ECGProcessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.gray = None
        self.binary = None
        self.gaus = None
        self.dx = None
        self.dy = None
        self.grid_confidence = None
        self.waveform = None
        self.times = None
        self.volts = None
        self.baseline = None
        self.interpolated_waveform = None
        self.roi_waveform = None
        self.signal = None
        self.peaks = None
        self.sampling_freq = None

    def calculate_sampling_frequency(self):
        """Calculate sampling frequency from grid calibration (dx = px/mm, speed = 25 mm/s)."""
        if self.dx is None:
            raise ValueError("Grid spacing not detected — run detect_grid_spacing() first.")
        self.sampling_freq = 25.0 * self.dx  # 25 mm/s * px/mm = px/s = Hz
        return self.sampling_freq

    def _adaptive_threshold(self, signal, fs):
        """Dynamic threshold calculation for Pan-Tompkins detector."""
        thresholds = {
            'peak': np.mean(signal) * 2,
            'noise': np.mean(signal) * 0.5,
            'rr_low': 0.3,   # 300 ms minimum RR
            'rr_high': 2.0,  # 2000 ms maximum RR
        }
        window = int(fs * 2)
        for i in range(0, len(signal), window):
            segment = signal[i:i + window]
            thresholds['peak'] = 0.875 * thresholds['peak'] + 0.125 * np.max(segment)
            thresholds['noise'] = 0.875 * thresholds['noise'] + 0.125 * np.median(segment)
        return thresholds

    def _find_qrs_peaks(self, signal, thresholds, fs):
        """Peak detection with physiological refractory-period constraint."""
        peaks, _ = find_peaks(
            signal,
            height=thresholds['peak'],
            distance=int(fs * thresholds['rr_low']),
        )
        valid_peaks = []
        prev_peak = -np.inf
        for peak in peaks:
            if (peak - prev_peak) / fs > thresholds['rr_low']:
                valid_peaks.append(peak)
                prev_peak = peak
        return np.array(valid_peaks)

    def _detect_r_peaks_simple(self, fs):
        """Scipy find_peaks with prominence and width filtering."""
        height_thresh = np.max(self.signal) * 0.05
        prominence_thresh = 1.5 * np.std(self.signal)
        initial_peaks, properties = find_peaks(
            self.signal,
            height=height_thresh,
            distance=int(fs * 0.3),  # 300 ms refractory period
            prominence=prominence_thresh,
            width=1,
        )
        if 'widths' in properties and len(properties['widths']) > 0:
            median_width = np.median(properties['widths'])
            valid = properties['widths'] >= 0.5 * median_width
            self.peaks = initial_peaks[valid]
        else:
            self.peaks = initial_peaks

    def _detect_r_peaks_pantompkins(self, fs):
        """Pan-Tompkins-inspired pipeline: bandpass → differentiate → square → integrate."""
        filtered = bandpass_filter(self.signal, 5, 15, fs, order=2)
        diff = np.diff(filtered, prepend=filtered[0])
        squared = diff ** 2
        window_size = max(1, int(fs * 0.15))
        integrated = np.convolve(squared, np.ones(window_size) / window_size, 'same')
        thresholds = self._adaptive_threshold(integrated, fs)
        self.peaks = self._find_qrs_peaks(integrated, thresholds, fs)

    def detect_r_peaks(self, algorithm='simple'):
        """
        Detect R-wave peaks in self.signal.

        Args:
            algorithm: 'simple' — scipy find_peaks with prominence filtering (default).
                       'pantompkins' — Pan-Tompkins-inspired bandpass/integration pipeline.
        """
        fs = self.calculate_sampling_frequency()
        if algorithm == 'pantompkins':
            self._detect_r_peaks_pantompkins(fs)
        else:
            self._detect_r_peaks_simple(fs)

        if self.peaks is not None and len(self.peaks) > 1:
            rr_intervals = np.diff(self.peaks) / fs
            hr = 60.0 / np.median(rr_intervals)
            if not (40 < hr < 180):
                logger.warning("Unusual heart rate detected: %.1f bpm", hr)

    def load_and_prepare_image(self):
        if not os.path.isfile(self.image_path):
            raise FileNotFoundError(f"Image not found: {self.image_path}")
        logger.info("Loading image: %s", self.image_path)
        self.gray, self.binary, self.gaus = load_and_preprocess_image(self.image_path)

    def detect_grid_spacing(self):
        try:
            self.dx, self.dy, self.grid_confidence = robust_grid_spacing(self.binary, debug=False)
            logger.info(
                "Grid spacing: dx=%.2f px/mm, dy=%.2f px/mm (confidence: %.2f)",
                self.dx, self.dy, self.grid_confidence,
            )
            if self.grid_confidence < 0.4:
                logger.warning(
                    "Low grid detection confidence (%.2f) — measurements may be inaccurate. "
                    "Check image quality or use manual calibration.",
                    self.grid_confidence,
                )
        except RuntimeError as e:
            logger.error("Grid detection failed: %s", e)
            raise

    def isolate_and_extract_waveform(self):
        self.waveform = isolate_waveform(self.binary, dx=self.dx, dy=self.dy, debug=False)
        self.times, self.volts, self.baseline = extract_ecg_signal(
            self.waveform, dx=self.dx, dy=self.dy, debug=False
        )

    def interpolate_waveform(self):
        contours, _ = cv2.findContours(self.waveform, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.interpolated_waveform = np.zeros_like(self.waveform)
        for contour in contours:
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            cv2.polylines(self.interpolated_waveform, [approx], isClosed=False, color=255, thickness=2)

    def select_roi(self):
        selector = ROISelector(self.binary)
        roi_coords = selector.get_roi()
        if roi_coords:
            x, y, w, h = roi_coords
            if w > 0 and h > 0:
                self.roi_waveform = self.waveform[y:y + h, x:x + w]
                cv2.imshow("Cropped ROI", self.roi_waveform)
                cv2.waitKey(0)
            else:
                logger.warning("Empty ROI selected")
        else:
            logger.warning("ROI selection cancelled")

    def extract_1d_signal(self, window_size=5):
        signal_1d = np.sum(self.roi_waveform, axis=0)
        self.signal = np.convolve(signal_1d, np.ones(window_size) / window_size, mode='same')

    def calculate_heart_rate(self):
        r_wave_distances = np.diff(self.peaks)
        large_boxes = r_wave_distances / self.dx
        heart_rates = 300 / large_boxes
        return r_wave_distances, large_boxes, heart_rates

    def plot_results(self):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.title('Original Waveform')
        plt.imshow(self.waveform, cmap='gray')
        plt.subplot(1, 2, 2)
        plt.title('Interpolated Waveform')
        plt.imshow(self.interpolated_waveform, cmap='gray')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(6, 4))
        plt.title("Selected ROI")
        plt.imshow(self.roi_waveform, cmap='gray')
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.plot(self.signal, label="Smoothed Signal")
        plt.plot(self.peaks, self.signal[self.peaks], "rx", label="R-wave Peaks")
        plt.title("R-wave Peak Detection")
        plt.legend()
        plt.show()

    def run_pipeline(self):
        self.load_and_prepare_image()
        self.detect_grid_spacing()
        self.isolate_and_extract_waveform()
        self.interpolate_waveform()
        self.select_roi()
        self.extract_1d_signal()
        self.detect_r_peaks()
        self.plot_results()

        r_wave_distances, large_boxes, heart_rates = self.calculate_heart_rate()
        logger.info("R-wave distances (px): %s", r_wave_distances)
        logger.info("Large boxes between R-waves: %s", large_boxes)
        logger.info("Heart rates (bpm): %s", heart_rates)
        avg_hr = relative_mode(heart_rates)
        logger.info("Average heart rate (mode): %.2f bpm", avg_hr)
        print(f"Average Heart Rate: {avg_hr:.1f} bpm")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    processor = ECGProcessor(image_path='images/sample3.png')
    try:
        processor.run_pipeline()
    except Exception as e:
        logger.exception("Pipeline failed: %s", e)
