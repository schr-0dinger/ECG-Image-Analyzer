import logging
import math
from typing import Tuple, Union

import cv2
import numpy as np
from deskew import determine_skew

__all__ = ['load_and_preprocess_image']

logger = logging.getLogger(__name__)


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
    """Load, deskew, blur, and binarize an ECG image.

    Returns:
        gray (np.ndarray): original grayscale image.
        thresh (np.ndarray): Otsu-binarized result after deskewing and blurring.
        blurred (np.ndarray): median-blurred deskewed image.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    angle = determine_skew(image)
    if angle is None:
        angle = 0.0
    logger.debug("Deskew angle: %.2f°", angle)

    deskewed_image = rotate(image, angle, (0, 0, 0))
    blurred = cv2.medianBlur(deskewed_image, 1)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return image, thresh, blurred