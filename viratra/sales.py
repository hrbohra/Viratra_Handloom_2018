"""Layer 9 - Sales correlation / prediction.

Correlates a saree's visual + catalog attributes (pattern class, fabric, region,
price) with historical units sold, and predicts expected demand for an incoming
design. A RandomForestRegressor over one-hot encoded categoricals - the mature,
well-documented scikit-learn recipe.
"""

import pickle

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from . import config


class SalesModel(object):
    def __init__(self):
        tax = config.load_taxonomy()
        self.patterns = [p["key"] for p in sorted(tax["patterns"], key=lambda p: p["id"])]
        self.fabrics = list(tax["fabrics"])
        self.regions = list(tax["regions"])
        self.model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)

    def _encode(self, saree):
        """One-hot pattern/fabric/region + normalised price -> feature vector."""
        vec = []
        vec += [1.0 if saree.get("pattern") == p else 0.0 for p in self.patterns]
        vec += [1.0 if saree.get("fabric") == f else 0.0 for f in self.fabrics]
        vec += [1.0 if saree.get("region") == r else 0.0 for r in self.regions]
        vec.append(float(saree.get("price", 0.0)) / 10000.0)
        return np.asarray(vec, dtype=np.float32)

    def fit(self, sarees, units):
        X = np.vstack([self._encode(s) for s in sarees])
        self.model.fit(X, np.asarray(units, dtype=np.float32))
        return self

    def predict(self, saree):
        return float(self.model.predict([self._encode(saree)])[0])

    def feature_importance(self):
        """Return importance grouped back to the human attribute names."""
        imp = self.model.feature_importances_
        names = (["pattern:" + p for p in self.patterns]
                 + ["fabric:" + f for f in self.fabrics]
                 + ["region:" + r for r in self.regions]
                 + ["price"])
        pairs = sorted(zip(names, imp), key=lambda t: t[1], reverse=True)
        return [{"attribute": n, "importance": round(float(v), 4)} for n, v in pairs]

    def save(self, path=None):
        path = path or config.SALES_MODEL_PATH
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def load(cls, path=None):
        path = path or config.SALES_MODEL_PATH
        with open(path, "rb") as fh:
            return pickle.load(fh)
