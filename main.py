import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import mode

from ecg_image_loader import load_and_preprocess_image
from grid_detection import robust_grid_spacing
from isolate_waveform import isolate_waveform
from waveform_extraction import extract_ecg_signal

class ECGProcessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.gray = None
        self.binary = None
        self.gaus = None
        self.dx = None
        self.dy = None
        self.waveform = None
        self.times = None
        self.volts = None
        self.baseline = None
        self.interpolated_waveform = None
        self.roi_waveform = None
        self.signal = None
        self.peaks = None

    def load_and_prepare_image(self):
        print(f"Loading image from: {self.image_path}")
        self.gray, self.binary, self.gaus = load_and_preprocess_image(self.image_path)

    def detect_grid_spacing(self):
        try:
            self.dx, self.dy = robust_grid_spacing(self.binary, debug=True)
            print(f"1 mm grid ≈ {self.dx:.2f}px horizontally, {self.dy:.2f}px vertically")
        except RuntimeError as e:
            print(f"Grid detection failed: {e}")
            raise

    def isolate_and_extract_waveform(self):
        self.waveform = isolate_waveform(self.binary, dx=self.dx, dy=self.dy, debug=True)
        self.times, self.volts, self.baseline = extract_ecg_signal(self.waveform, dx=self.dx, dy=self.dy, debug=True)

    def interpolate_waveform(self):
        contours, _ = cv2.findContours(self.waveform, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.interpolated_waveform = np.zeros_like(self.waveform)
        for contour in contours:
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            cv2.polylines(self.interpolated_waveform, [approx], isClosed=False, color=255, thickness=2)

    def select_roi(self):
        color_binary = cv2.cvtColor(self.binary, cv2.COLOR_GRAY2BGR)
        h, w = self.binary.shape
        cv2.line(color_binary, (0, h//2), (w, h//2), (0, 255, 0), 1)
        cv2.line(color_binary, (w//2, 0), (w//2, h), (0, 255, 0), 1)

        print("Please select the region corresponding to the desired lead (e.g., Lead II).")
        roi = cv2.selectROI("Select ROI", color_binary, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow("Select ROI")

        x, y, w, h = roi
        if w == 0 or h == 0:
            raise ValueError("Invalid ROI selected.")
        self.roi_waveform = self.waveform[y:y+h, x:x+w]

        if self.roi_waveform.size == 0:
            raise ValueError("The selected ROI is empty.")

    def extract_1d_signal(self, window_size=5):
        signal_1d = np.sum(self.roi_waveform, axis=0)
        self.signal = np.convolve(signal_1d, np.ones(window_size)/window_size, mode='same')

    def detect_r_peaks(self):
        height_thresh = np.max(self.signal) * 0.05
        signal_std = np.std(self.signal)
        prominence_thresh = 1.5 * signal_std

        initial_peaks, properties = find_peaks(
            self.signal, height=height_thresh, distance=self.dx*2, prominence=prominence_thresh, width=1
        )

        if 'widths' in properties and len(properties['widths']) > 0:
            median_width = np.median(properties['widths'])
            valid_indices = properties['widths'] >= 0.5 * median_width
            self.peaks = initial_peaks[valid_indices]
        else:
            self.peaks = initial_peaks

    def calculate_heart_rate(self):
        r_wave_distances = np.diff(self.peaks)
        large_boxes = r_wave_distances / self.dx
        heart_rates = 300 / large_boxes
        return r_wave_distances, large_boxes, heart_rates

    def plot_results(self):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1); plt.title('Original Waveform'); plt.imshow(self.waveform, cmap='gray')
        plt.subplot(1, 2, 2); plt.title('Interpolated Waveform'); plt.imshow(self.interpolated_waveform, cmap='gray')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(6, 4))
        plt.title("Selected ROI (Lead II)")
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

        print("R-wave distances (in pixels):", r_wave_distances)
        print("Number of small boxes between R-waves:", large_boxes)
        print("Heart rates (bpm):", heart_rates)

        mode_heart_rate = mode(heart_rates).mode
        avg_mode_heart_rate = np.mean(mode_heart_rate)
        print(f"Average of Mode Heart Rate: {avg_mode_heart_rate:.2f} bpm")


if __name__ == "__main__":
    processor = ECGProcessor(image_path='images/ed1.png')
    try:
        processor.run_pipeline()
    except Exception as e:
        print(f"An error occurred: {e}")
