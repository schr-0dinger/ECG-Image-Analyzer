from ecg_image_loader import load_and_preprocess_image
from grid_detection import robust_grid_spacing
from isolate_waveform import isolate_waveform
from waveform_extraction import extract_ecg_signal
import cv2

def main():
    # Load and preprocess the image
    path = 'images/sample3.png'  # Replace with your image path
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

    # After waveform extraction
    waveform = cv2.dilate(waveform, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations=1)

    # Optionally, display images
    cv2.imshow('Gray Image', gray)
    cv2.imshow('Binary Image', binary)
    cv2.imshow('Waveform Only', waveform)

    # Wait for a key press and close windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()