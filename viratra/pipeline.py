"""End-to-end design-intelligence pipeline.

The Engine ties the layers together: preprocess -> colour -> pattern -> deep
fingerprint -> similarity search, plus optional sales prediction. It is the
single entry point used by both the CLI and the Electron main process, and it
degrades gracefully when a model artifact has not been built yet.
"""

import os

from . import (color, config, embeddings, patterns, preprocess, search)
from .db import Database


class Engine(object):
    def __init__(self, db_path=None):
        self.labels = config.pattern_labels()
        self.db = Database(db_path)
        self._clf = None
        self._index = None
        self._sales = None

    # ---- lazy model loading ---------------------------------------------
    @property
    def classifier(self):
        if self._clf is None and os.path.exists(config.CLASSIFIER_PATH):
            self._clf = patterns.PatternClassifier.load(config.CLASSIFIER_PATH)
        return self._clf

    @property
    def index(self):
        if self._index is None and os.path.exists(config.INDEX_PATH + ".meta.npy"):
            self._index = search.FingerprintIndex.load()
        return self._index

    @property
    def sales(self):
        if self._sales is None and os.path.exists(config.SALES_MODEL_PATH):
            from .sales import SalesModel
            self._sales = SalesModel.load()
        return self._sales

    # ---- core operations -------------------------------------------------
    def classify(self, prep):
        """Return pattern prediction dict, or None if no classifier is built."""
        clf = self.classifier
        if clf is None:
            return None
        feats = patterns.pattern_features(prep)
        scores = clf.predict_proba(feats)
        top = max(scores, key=scores.get)
        return {
            "key": top,
            "label": self.labels.get(top, top),
            "confidence": scores[top],
            "scores": scores,
        }

    def fingerprint(self, prep):
        """The deep embedding vector used for similarity search."""
        return embeddings.embed(prep)

    def similar(self, prep, k=6, exclude_id=None):
        """Return up to k similar catalog sarees for a prepared image."""
        idx = self.index
        if idx is None:
            return []
        vec = self.fingerprint(prep)
        hits = idx.search(vec, k=k + (1 if exclude_id is not None else 0))
        out = []
        for saree_id, sim in hits:
            if exclude_id is not None and saree_id == exclude_id:
                continue
            rec = self.db.get_saree(saree_id)
            if rec:
                rec["similarity"] = sim
                out.append(rec)
            if len(out) >= k:
                break
        return out

    def analyze(self, image_path, k=6):
        """Full analysis of an arbitrary photo (not necessarily in catalog)."""
        prep = preprocess.prepare(image_path)
        result = {
            "image": os.path.abspath(image_path),
            "colors": color.dominant_colors(prep["bgr"], k=5),
            "pattern": self.classify(prep),
            "embedding_backend": embeddings.backend_name(),
            "search_backend": search.backend_name(),
            "similar": self.similar(prep, k=k),
        }
        return result

    def close(self):
        self.db.close()
