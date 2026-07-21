"""Layer 5 - Colour intelligence.

Dominant-colour extraction via k-means (the canonical PyImageSearch recipe) and
an HSV colour histogram that becomes part of the design fingerprint.
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans


def dominant_colors(bgr, k=5):
    """Return the k dominant colours of an image, most-prominent first.

    Each entry is a dict with the RGB triplet, a #rrggbb hex string and the
    fraction of pixels assigned to that cluster. Mirrors the standard
    reshape-to-pixels then KMeans approach.
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pixels = rgb.reshape((-1, 3)).astype(np.float64)

    km = KMeans(n_clusters=k, n_init=4, random_state=42)
    labels = km.fit_predict(pixels)

    counts = np.bincount(labels, minlength=k).astype(np.float64)
    fractions = counts / counts.sum()

    order = np.argsort(fractions)[::-1]
    result = []
    for idx in order:
        r, g, b = (int(round(v)) for v in km.cluster_centers_[idx])
        result.append({
            "rgb": [r, g, b],
            "hex": "#{:02x}{:02x}{:02x}".format(r, g, b),
            "fraction": round(float(fractions[idx]), 4),
        })
    return result


def color_histogram(hsv, bins=(8, 8, 8)):
    """Normalised 3D HSV colour histogram flattened to a feature vector."""
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                        [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten().astype(np.float32)
