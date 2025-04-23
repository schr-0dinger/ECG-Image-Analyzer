import cv2
import math
import numpy as np
from typing import Tuple, Union
from deskew import determine_skew


def rotate(image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]) -> np.ndarray:
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)

def load_and_preprocess_image(image_path):

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Greyscale         // B&W

    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded. Please check the path and file format.")

    # Deskew the image first
    angle = determine_skew(image)
    if angle is None:
        angle = 0  # Default to no rotation if skew cannot be determined

    deskewed_image = rotate(image, angle, (0,0,0))
    cv2.imshow('Deskewed Image', deskewed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    blurred = cv2.medianBlur(deskewed_image, 1)     # Gaussian blur     // To reduce noise

                                                    # Otsu binarization // Highlights waveform
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 

    # Adaptive thresholding // Highlights waveform
    # thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 9)

    return image, thresh, blurred