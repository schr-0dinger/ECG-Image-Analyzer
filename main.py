from ecg_image_loader import load_and_preprocess_image
from grid_detection import robust_grid_spacing
from isolate_waveform import isolate_waveform
from waveform_extraction import extract_ecg_signal
import numpy as np
import cv2
import matplotlib.pyplot as plt

def main():
    # Load and preprocess the image
    path = 'images/sample.png'  # Replace with your image path
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

if __name__ == "__main__":
    main()