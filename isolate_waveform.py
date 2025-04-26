import cv2
import numpy as np

def isolate_waveform(image, dx=None, dy=None, debug=True):
    """
    Suppress the ECG grid and enhance the waveform. Assumes input is grayscale or binary.
    Returns a binary image where the ECG trace is white on black.
    """
    # If image is not already binary, threshold it
    if image.dtype != np.uint8 or np.max(image) > 1:
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    else:
        binary = image.copy()

    # Use smaller kernels to avoid erasing the trace
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ( 33, 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 33))
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1)

    grid_mask = cv2.bitwise_or(horizontal_lines, vertical_lines)
    waveform = cv2.bitwise_and(binary, cv2.bitwise_not(grid_mask))

    if debug:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,4))
        plt.subplot(1,4,1); plt.title('Binary'); plt.imshow(binary, cmap='gray')
        plt.subplot(1,4,2); plt.title('Horizontal Grid'); plt.imshow(horizontal_lines, cmap='gray')
        plt.subplot(1,4,3); plt.title('Vertical Grid'); plt.imshow(vertical_lines, cmap='gray')
        plt.subplot(1,4,4); plt.title('Waveform'); plt.imshow(waveform, cmap='gray')
        plt.tight_layout()
        plt.show()

    return waveform