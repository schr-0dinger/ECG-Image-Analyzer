from ecg_image_loader import load_and_preprocess_image
from grid_detection import robust_grid_spacing
from isolate_waveform import isolate_waveform
from waveform_extraction import extract_ecg_signal
from scipy.signal import find_peaks
import numpy as np
import cv2
import matplotlib.pyplot as plt

def main():
    # Load and preprocess the image
    path = 'images/sample4.png'  # Replace with your image path
    print(f"Loading image from: {path}")
    gray, binary, gaus = load_and_preprocess_image(path)

    # Determine grid spacing (pixels per mm)
    try:
        dx, dy = robust_grid_spacing(binary, debug=True)
    except RuntimeError as e:
        print(f"Grid detection failed: {e}")
        return
    print(f"1 mm grid ≈ {dx:.2f}px horizontally, {dy:.2f}px vertically")

    # Isolate the ECG waveform (remove grid) from binary iamge
    waveform = isolate_waveform(binary, dx=dx, dy=dy, debug=True)

    # Extract 1D ECG signal in real-world units
    times, volts, baseline = extract_ecg_signal(waveform, dx, dy, debug=True)

    # After waveform isolation
    # Detect contours of the waveform
    contours, _ = cv2.findContours(waveform, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank canvas to draw the interpolated waveform
    interpolated_waveform = np.zeros_like(waveform)

    # Iterate through each contour and interpolate gaps
    for contour in contours:
        # Approximate the contour to reduce noise
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Draw the approximated contour on the blank canvas
        cv2.polylines(interpolated_waveform, [approx], isClosed=False, color=255, thickness=2)

    # Optionally, display the result
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1); plt.title('Original Waveform'); plt.imshow(waveform, cmap='gray')
    plt.subplot(1, 2, 2); plt.title('Interpolated Waveform'); plt.imshow(interpolated_waveform, cmap='gray')
    plt.tight_layout()
    plt.show()

    # Convert binary image to a color image for better visibility
    color_binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    # Add a grid or crosshair (optional)
    cv2.line(color_binary, (0, binary.shape[0] // 2), (binary.shape[1], binary.shape[0] // 2), (0, 255, 0), 1)
    cv2.line(color_binary, (binary.shape[1] // 2, 0), (binary.shape[1] // 2, binary.shape[0]), (0, 255, 0), 1)

    # Prompt the user to select the ROI
    print("Please select the region corresponding to the desired lead (e.g., Lead II).")
    roi = cv2.selectROI("Select ROI", color_binary, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select ROI")

    # Crop the selected ROI
    x, y, w, h = roi
    roi_waveform = waveform[int(y):int(y+h), int(x):int(x+w)]

    # Display the cropped ROI for confirmation
    plt.figure(figsize=(6, 4))
    plt.title("Selected ROI (Lead II)")
    plt.imshow(roi_waveform, cmap='gray')
    plt.show()

    # Convert the binary image into a 1D signal directly
    signal_1d = np.sum(roi_waveform, axis=0)  # Sum along the vertical axis

    # Smooth the signal using a moving average
    window_size = 5  # Adjust the window size as needed
    smoothed_signal = np.convolve(signal_1d, np.ones(window_size)/window_size, mode='same')

    # Dynamically set the height threshold
    height_threshold = np.max(smoothed_signal) * 0.05  # Adjust the multiplier as needed

    # Estimate signal noise level
    signal_std = np.std(smoothed_signal)

    # Set prominence dynamically based on noise
    prominence_threshold = 1.5 * signal_std  # can adjust 1.5 to 2.0 based on sensitivity

    # First pass: detect all possible peaks
    initial_peaks, properties = find_peaks(
    smoothed_signal,
    height=height_threshold,
    distance=dx*2,
    prominence=prominence_threshold,
    width=1)

    # Adaptive width filtering
    # Calculate median width among initial detected peaks
    if 'widths' in properties and len(properties['widths']) > 0:
        median_width = np.median(properties['widths'])
        width_threshold = 0.5 * median_width  # keep only peaks wider than 50% of median width
    
        # Keep only valid peaks
        valid_indices = properties['widths'] >= width_threshold
        final_peaks = initial_peaks[valid_indices]
    else:
        final_peaks = initial_peaks  # fallback if no width info available

    # Plot the detected peaks
    plt.figure(figsize=(10, 4))
    plt.plot(smoothed_signal, label="Smoothed Signal")
    plt.plot(final_peaks, smoothed_signal[final_peaks], "rx", label="R-wave Peaks")
    plt.title("Improved R-wave Peak Detection")
    plt.legend()
    plt.show()

    # Calculate distances between consecutive R-wave peaks (in pixels)
    r_wave_distances = np.diff(final_peaks)

    # Convert pixel distances to the number of small boxes (1 mm = dx pixels)
    large_boxes = r_wave_distances / dx

    # Calculate heart rate for each R-R interval
    heart_rates = 300 / large_boxes

    # Display the results
    print("R-wave distances (in pixels):", r_wave_distances)
    print("Number of small boxes between R-waves:", large_boxes)
    print("Heart rates (bpm):", heart_rates)

    # Calculate the average heart rate
    average_heart_rate = np.mean(heart_rates)
    print(f"Average Heart Rate: {average_heart_rate:.2f} bpm")

if __name__ == "__main__":
    main()