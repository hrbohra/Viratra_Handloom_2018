"""Layer 3 - OpenCV image preprocessing.

Normalises in-shop photographs into a canonical form and exposes the classic
preprocessing primitives (grayscale, threshold, Canny, contours) that the
higher layers build on. Mirrors the official OpenCV-Python tutorials: apply a
threshold or Canny edge map before finding contours.
"""

import cv2
import numpy as np

from . import config


def load_bgr(path):
    """Load an image as BGR uint8, raising a clear error if it is missing."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError("Could not read image: {}".format(path))
    return img


def normalize(img, size=None):
    """Resize to the canonical working size using area interpolation."""
    size = size or config.IMAGE_SIZE
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def to_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def threshold(gray, block=35, c=5):
    """Adaptive threshold - robust to the uneven lighting of shop photos."""
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block, c,
    )


def edges(gray, low=50, high=150):
    """Canny edge map. A light Gaussian blur first tames JPEG noise."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blurred, low, high)


def contours(gray):
    """Return external contours found on the Canny edge map.

    Following the OpenCV tutorial guidance, edges are computed before
    findContours. Returns the list of contours sorted largest-first.
    """
    edged = edges(gray)
    found = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.findContours returns (contours, hierarchy) on OpenCV 4/5 and
    # (image, contours, hierarchy) on OpenCV 3 - handle both.
    cnts = found[0] if len(found) == 2 else found[1]
    return sorted(cnts, key=cv2.contourArea, reverse=True)


def prepare(path):
    """Full normalisation used by every downstream layer.

    Returns a dict with the resized BGR, grayscale and HSV representations so
    later stages do not each re-decode the file.
    """
    bgr = normalize(load_bgr(path))
    return {
        "bgr": bgr,
        "gray": to_gray(bgr),
        "hsv": to_hsv(bgr),
    }
