"""Engine smoke tests.

Fast unit checks that never require the trained artifacts, plus a couple of
integration checks that activate only once the models have been built
(scripts/build_all.py). Run with:  python -m pytest -q
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viratra import color, config, embeddings, patterns, preprocess, texture   # noqa: E402

SAMPLE = os.path.join(config.SAMPLES_DIR, "temple", "temple_00.png")
HAS_SAMPLES = os.path.exists(SAMPLE)
HAS_CLF = os.path.exists(config.CLASSIFIER_PATH)
HAS_INDEX = os.path.exists(config.INDEX_PATH + ".meta.npy")


# ---- taxonomy ------------------------------------------------------------
def test_taxonomy_has_eight_patterns():
    keys = config.pattern_keys()
    assert len(keys) == 8
    assert "temple" in keys and "paisley" in keys


# ---- preprocessing -------------------------------------------------------
@pytest.mark.skipif(not HAS_SAMPLES, reason="samples not generated")
def test_prepare_shapes():
    prep = preprocess.prepare(SAMPLE)
    assert prep["bgr"].shape == (config.IMAGE_SIZE[1], config.IMAGE_SIZE[0], 3)
    assert prep["gray"].ndim == 2


@pytest.mark.skipif(not HAS_SAMPLES, reason="samples not generated")
def test_color_and_texture_vectors():
    prep = preprocess.prepare(SAMPLE)
    cols = color.dominant_colors(prep["bgr"], k=5)
    assert len(cols) == 5
    assert cols[0]["hex"].startswith("#") and len(cols[0]["hex"]) == 7
    tex = texture.texture_vector(prep["gray"])
    assert tex.shape[0] == texture.LBP_BINS + 2 * len(texture.GABOR_FREQS) * len(texture.GABOR_THETAS)


@pytest.mark.skipif(not HAS_SAMPLES, reason="samples not generated")
def test_embedding_is_l2_normalised():
    prep = preprocess.prepare(SAMPLE)
    vec = embeddings.embed(prep)
    assert vec.dtype == np.float32
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-4


# ---- integration (needs built artifacts) ---------------------------------
@pytest.mark.skipif(not (HAS_SAMPLES and HAS_CLF), reason="classifier not built")
def test_classifier_predicts_temple():
    clf = patterns.PatternClassifier.load(config.CLASSIFIER_PATH)
    feats = patterns.pattern_features(preprocess.prepare(SAMPLE))
    assert clf.predict_label(feats) == "temple"


@pytest.mark.skipif(not (HAS_SAMPLES and HAS_INDEX), reason="index not built")
def test_search_returns_self_first():
    from viratra.pipeline import Engine
    eng = Engine()
    try:
        prep = preprocess.prepare(SAMPLE)
        hits = eng.index.search(eng.fingerprint(prep), k=1)
        assert hits and hits[0][1] > 0.99   # near-perfect self similarity
    finally:
        eng.close()
