import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
from scipy.stats import mode

from ecg_image_loader import load_and_preprocess_image
from grid_detection import robust_grid_spacing
from isolate_waveform import isolate_waveform
from waveform_extraction import extract_ecg_signal
from roi_selector import ROISelector

def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

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
        self.sampling_freq = None

    def calculate_sampling_frequency(self):
        """Calculate sampling frequency from grid calibration (dx)"""
        if self.dx is None:
            raise ValueError("Grid spacing not detected")
        self.sampling_freq = 25 * self.dx  # 25 mm/s * pixels/mm
        return self.sampling_freq

    def _adaptive_threshold(self, signal, fs):
        """Dynamic threshold calculation"""
        thresholds = {
            'peak': np.mean(signal) * 2,
            'noise': np.mean(signal) * 0.5,
            'rr_low': 0.3,  # 300ms refractory
            'rr_high': 2.0  # 2000ms timeout
        }
        window = int(fs * 2)
        for i in range(0, len(signal), window):
            segment = signal[i:i+window]
            thresholds['peak'] = 0.875 * thresholds['peak'] + 0.125 * np.max(segment)
            thresholds['noise'] = 0.875 * thresholds['noise'] + 0.125 * np.median(segment)
        return thresholds

    def _find_qrs_peaks(self, signal, thresholds, fs):
        """Peak detection with physiological constraints"""
        peaks, _ = find_peaks(
            signal,
            height=thresholds['peak'],
            distance=int(fs * thresholds['rr_low'])
        )
        
        valid_peaks = []
        prev_peak = -np.inf
        for peak in peaks:
            rr_interval = (peak - prev_peak)/fs
            if rr_interval > thresholds['rr_low']:
                valid_peaks.append(peak)
                prev_peak = peak
        return np.array(valid_peaks)

    def detect_r_peaks(self):
        """Main detection method using Pan-Tompkins algorithm"""
        # Calculate sampling frequency from grid calibration
        fs = self.calculate_sampling_frequency()
        
        # Validate signal quality
        if np.std(self.signal) < 0.1 * np.max(self.signal):
            raise ValueError("Signal too noisy for reliable detection")
            
        # 1. Bandpass filter
        filtered = bandpass_filter(self.signal, 5, 15, fs, order=2)
        
        # 2. Differentiation and squaringc
        diff = np.diff(filtered, prepend=filtered[0])
        squared = diff ** 2
        
        # 3. Moving integration
        window_size = int(fs * 0.15)
        integrated = np.convolve(squared, np.ones(window_size)/window_size, 'same')
        
        # 4. Adaptive thresholding
        thresholds = self._adaptive_threshold(integrated, fs)
        
        # 5. Peak detection
        self.peaks = self._find_qrs_peaks(integrated, thresholds, fs)
        
        # Validate heart rate
        rr_intervals = np.diff(self.peaks)/fs
        hr = 60 / np.median(rr_intervals) if len(rr_intervals) > 0 else 0
        if not (40 < hr < 180):
            print(f"Warning: Unusual heart rate detected: {hr:.1f} bpm")
    
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
        """ TODO: batch mode
        color_binary = cv2.cvtColor(self.binary, cv2.COLOR_GRAY2BGR)
        h, w = self.binary.shape
        cv2.line(color_binary, (0, h//2), (w, h//2), (0, 255, 0), 1)
        cv2.line(color_binary, (w//2, 0), (w//2, h), (0, 255, 0), 1)

        print("Please select the region corresponding to the desired lead (e.g., Lead II).")
        roi = cv2.selectROI("Select ROI", color_binary, showCrosshair=False, fromCenter=False)
        cv2.destroyWindow("Select ROI")

        x, y, w, h = roi
        if w == 0 or h == 0:
            raise ValueError("Invalid ROI selected.")
        if w > 0 & h > 0:
            self.roi_waveform = self.waveform[y:y+h, x:x+w]
        else:
            print("No ROI selected")    

        if self.roi_waveform.size == 0:
            raise ValueError("The selected ROI is empty.") """

        selector = ROISelector(self.binary)
        roi_coords = selector.get_roi()

        if roi_coords:
            x, y, w, h = roi_coords
            if w > 0 and h > 0:
                self.roi_waveform = self.waveform[y:y+h, x:x+w]
                cv2.imshow("Cropped ROI", self.roi_waveform)
                cv2.waitKey(0)
            else:
                print("Empty ROI selected")
        else:
            print("ROI selection cancelled")            

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
        print("Stage 1")
        self.detect_grid_spacing()
        self.interpolate_waveform()
        self.select_roi()
        self.extract_1d_signal()
        self.detect_r_peaks()
        self.plot_results()

        r_wave_distances, large_boxes, heart_rates = self.calculate_heart_rate()

        print("R-wave distances (in pixels):", r_wave_distances)
        print("Number of large boxes between R-waves:", large_boxes)
        print("Heart rates (bpm):", heart_rates)

        def relative_mode(heart_rates, threshold=5):
            groups = []
            for hr in heart_rates:
                group = [x for x in heart_rates if abs(x - hr) <= threshold]
                if len(group) > 1:
                    groups.append(group)
            if groups:
                largest_group = max(groups, key=len)
                return np.mean(largest_group)
            else:
                return np.mean(heart_rates)

        avg_mode_heart_rate = relative_mode(heart_rates)
        print(f"Average heart rate (mode): {avg_mode_heart_rate:.2f} bpm")

if __name__ == "__main__":
    processor = ECGProcessor(image_path='images/sample3.png')
    try:
        processor.run_pipeline()
    except Exception as e:
        print(f"An error occurred: {e}")
        # print(cv2.getBuildInformation())
