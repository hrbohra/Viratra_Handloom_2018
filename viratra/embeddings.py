"""Layer 7 - Deep-feature "design fingerprint" extraction.

The original 2018 build pulled 512-d bottleneck features from a headless VGG16
(Keras / TensorFlow, ImageNet weights, global-average pooling). That remains the
preferred backend and is used automatically when Keras is importable.

When Keras/TensorFlow is not installed, the engine falls back to a
deterministic classical descriptor (HSV colour histogram + LBP + Gabor + HOG).
Both backends return an L2-normalised float32 vector, so every downstream
consumer (FAISS index, similarity search) is agnostic to which one produced it.
"""

import numpy as np

from . import color, texture

# ---- Backend detection ---------------------------------------------------
_KERAS = None
_KERAS_MODEL = None


def _try_keras():
    """Import Keras/VGG16 lazily; return the model or None if unavailable."""
    global _KERAS, _KERAS_MODEL
    if _KERAS is not None:
        return _KERAS_MODEL
    try:
        from keras.applications.vgg16 import VGG16, preprocess_input  # noqa
        _KERAS_MODEL = {
            "model": VGG16(weights="imagenet", include_top=False, pooling="avg"),
            "preprocess": preprocess_input,
        }
        _KERAS = True
    except Exception:
        _KERAS = False
        _KERAS_MODEL = None
    return _KERAS_MODEL


def backend_name():
    return "vgg16-keras" if _try_keras() else "classical-hybrid"


# ---- Public API ----------------------------------------------------------
def embed(prep):
    """Return the design-fingerprint vector for a prepared image dict.

    `prep` is the dict returned by preprocess.prepare(): bgr / gray / hsv.
    """
    model = _try_keras()
    if model is not None:
        vec = _embed_vgg16(prep["bgr"], model)
    else:
        vec = _embed_classical(prep)
    return _l2(vec.astype(np.float32))


def _embed_vgg16(bgr, model):
    import cv2
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64)
    batch = model["preprocess"](np.expand_dims(rgb, axis=0))
    feats = model["model"].predict(batch, verbose=0)
    return feats.reshape(-1)


def _embed_classical(prep):
    from skimage.feature import hog

    col = color.color_histogram(prep["hsv"])          # 512
    tex = texture.texture_vector(prep["gray"])        # 42
    shape = hog(
        prep["gray"],
        orientations=9,
        pixels_per_cell=(32, 32),
        cells_per_block=(2, 2),
        feature_vector=True,
    ).astype(np.float32)                               # 1764
    return np.concatenate([col, tex, shape])


def _l2(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec
