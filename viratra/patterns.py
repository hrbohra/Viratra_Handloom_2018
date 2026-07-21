"""Layer 4 - Pattern recognition.

Assembles hand-engineered features (colour histogram + LBP/Gabor texture +
coarse shape statistics) and trains a classifier over the eight saree pattern
classes. The default estimator is a RandomForest; a linear SVM is available via
the `estimator` argument for the interpretable v1 recipe described in the
project notes.
"""

import pickle

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from . import color, preprocess, texture


def pattern_features(prep):
    """Feature vector used by the pattern classifier.

    colour histogram (512) + texture signature (42) + shape stats (2).
    """
    col = color.color_histogram(prep["hsv"])
    tex = texture.texture_vector(prep["gray"])

    edged = preprocess.edges(prep["gray"])
    edge_density = np.float32(edged.mean() / 255.0)
    cnts = preprocess.contours(prep["gray"])
    contour_count = np.float32(min(len(cnts), 500) / 500.0)

    shape = np.asarray([edge_density, contour_count], dtype=np.float32)
    return np.concatenate([col, tex, shape])


def features_for_path(path):
    return pattern_features(preprocess.prepare(path))


class PatternClassifier(object):
    """Thin wrapper around a scaled sklearn estimator with save/load."""

    def __init__(self, classes, estimator="rf"):
        self.classes = list(classes)
        self.estimator = estimator
        if estimator == "svm":
            clf = LinearSVC(C=1.0)
        else:
            clf = RandomForestClassifier(
                n_estimators=200, max_depth=None, random_state=42, n_jobs=-1,
            )
        self.pipeline = Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)

    def predict_label(self, feature_vec):
        idx = int(self.pipeline.predict([feature_vec])[0])
        return self.classes[idx]

    def predict_proba(self, feature_vec):
        """Return a {class_key: probability} dict.

        RandomForest exposes predict_proba directly; LinearSVC does not, so its
        decision-function margins are softmax-normalised into pseudo-scores.
        """
        clf = self.pipeline.named_steps["clf"]
        Xs = self.pipeline.named_steps["scaler"].transform([feature_vec])
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(Xs)[0]
        else:
            margins = clf.decision_function(Xs)[0]
            e = np.exp(margins - np.max(margins))
            probs = e / e.sum()
        return {self.classes[i]: round(float(p), 4) for i, p in enumerate(probs)}

    def save(self, path):
        with open(path, "wb") as fh:
            pickle.dump({
                "classes": self.classes,
                "estimator": self.estimator,
                "pipeline": self.pipeline,
            }, fh)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        obj = cls(state["classes"], estimator=state["estimator"])
        obj.pipeline = state["pipeline"]
        return obj
