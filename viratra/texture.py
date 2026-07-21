"""Layer 6 - Texture analysis.

Local Binary Patterns (uniform) and a Gabor filter bank, both from
scikit-image. LBP captures fine motif texture (butta density, weave grain);
Gabor energies capture directional weave structure (silk vs cotton vs brocade).
"""

import numpy as np
from skimage.feature import local_binary_pattern
from skimage.filters import gabor

# Uniform LBP with P=24, R=3 - the configuration recommended in the
# PyImageSearch LBP tutorial for texture classification.
LBP_POINTS = 24
LBP_RADIUS = 3
LBP_BINS = LBP_POINTS + 2  # uniform patterns -> P + 2 bins

# Gabor bank: four orientations x two frequencies.
GABOR_FREQS = (0.1, 0.3)
GABOR_THETAS = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)


def lbp_histogram(gray):
    """Normalised histogram of uniform LBP codes."""
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=LBP_BINS, range=(0, LBP_BINS))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-7)
    return hist


def gabor_energy(gray):
    """Mean and variance of the magnitude response for each Gabor filter.

    Returns a flat vector of length 2 * n_freqs * n_thetas.
    """
    g = gray.astype(np.float64) / 255.0
    feats = []
    for freq in GABOR_FREQS:
        for theta in GABOR_THETAS:
            real, imag = gabor(g, frequency=freq, theta=theta)
            mag = np.sqrt(real ** 2 + imag ** 2)
            feats.append(mag.mean())
            feats.append(mag.var())
    return np.asarray(feats, dtype=np.float32)


def texture_vector(gray):
    """Concatenated LBP histogram + Gabor energy signature."""
    return np.concatenate([lbp_histogram(gray), gabor_energy(gray)])
