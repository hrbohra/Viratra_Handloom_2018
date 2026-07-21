"""Train the Layer-4 saree pattern classifier.

Walks the labelled sample folders, extracts the hand-engineered feature vector
for every swatch, fits a scaled RandomForest (or linear SVM) and reports a
held-out classification report and confusion matrix before saving the model.

Usage:
    python scripts/train_classifier.py --estimator rf
"""

import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viratra import config, patterns, preprocess        # noqa: E402


def load_dataset():
    keys = config.pattern_keys()
    key_to_id = {k: i for i, k in enumerate(keys)}
    X, y, paths = [], [], []
    for key in keys:
        sample_dir = os.path.join(config.SAMPLES_DIR, key)
        if not os.path.isdir(sample_dir):
            continue
        for fname in sorted(os.listdir(sample_dir)):
            if not fname.endswith(".png"):
                continue
            path = os.path.join(sample_dir, fname)
            X.append(patterns.pattern_features(preprocess.prepare(path)))
            y.append(key_to_id[key])
            paths.append(path)
    return np.asarray(X), np.asarray(y), keys


def main(estimator):
    config.ensure_dirs()
    print("Extracting features ...")
    X, y, keys = load_dataset()
    print("  dataset: {} samples, {} features, {} classes".format(
        X.shape[0], X.shape[1], len(keys)))

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42)

    clf = patterns.PatternClassifier(keys, estimator=estimator).fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    print("\n=== Held-out classification report ({}) ===".format(estimator))
    print(classification_report(y_te, y_pred, target_names=keys, digits=3))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_te, y_pred))

    # Refit on the full dataset before persisting.
    clf = patterns.PatternClassifier(keys, estimator=estimator).fit(X, y)
    clf.save(config.CLASSIFIER_PATH)

    report = classification_report(y_te, y_pred, target_names=keys,
                                   output_dict=True, digits=3)
    with open(os.path.join(config.MODELS_DIR, "classifier_metrics.json"), "w") as fh:
        json.dump({"estimator": estimator, "accuracy": report["accuracy"],
                   "classes": keys}, fh, indent=2)
    print("\nSaved classifier -> {}".format(config.CLASSIFIER_PATH))
    print("Held-out accuracy: {:.1%}".format(report["accuracy"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimator", choices=["rf", "svm"], default="rf")
    args = ap.parse_args()
    main(args.estimator)
