"""Layer 8 - Similar-design search.

Indexes the design fingerprints and returns nearest neighbours for
"find sarees that look like this one". Uses FAISS when available and falls
back to scikit-learn NearestNeighbors, which is all a catalog of a few thousand
designs needs.

Fingerprints are L2-normalised, so inner-product search is cosine similarity.
"""

import os

import numpy as np

from . import config

try:
    import faiss
    _HAVE_FAISS = True
except Exception:  # pragma: no cover - environment dependent
    _HAVE_FAISS = False


def backend_name():
    return "faiss" if _HAVE_FAISS else "sklearn-nn"


class FingerprintIndex(object):
    """A searchable index of (catalog_id -> fingerprint) pairs."""

    def __init__(self, dim):
        self.dim = int(dim)
        self.ids = np.zeros((0,), dtype=np.int64)
        if _HAVE_FAISS:
            self._index = faiss.IndexFlatIP(self.dim)
            self._matrix = None
        else:
            self._index = None
            self._matrix = np.zeros((0, self.dim), dtype=np.float32)

    def add(self, vectors, ids):
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        ids = np.asarray(ids, dtype=np.int64)
        if vectors.shape[1] != self.dim:
            raise ValueError("expected dim {}, got {}".format(self.dim, vectors.shape[1]))
        self.ids = np.concatenate([self.ids, ids])
        if _HAVE_FAISS:
            self._index.add(vectors)
        else:
            self._matrix = np.vstack([self._matrix, vectors])

    def search(self, vector, k=5):
        """Return up to k (catalog_id, similarity) pairs, best first."""
        q = np.ascontiguousarray([vector], dtype=np.float32)
        n = len(self.ids)
        if n == 0:
            return []
        k = min(k, n)
        if _HAVE_FAISS:
            scores, idx = self._index.search(q, k)
            scores, idx = scores[0], idx[0]
        else:
            sims = (self._matrix @ q[0])  # cosine, vectors already normalised
            idx = np.argsort(sims)[::-1][:k]
            scores = sims[idx]
        out = []
        for j, s in zip(idx, scores):
            if j < 0:
                continue
            out.append((int(self.ids[j]), round(float(s), 4)))
        return out

    # ---- persistence -----------------------------------------------------
    def save(self, index_path=None, ids_path=None):
        index_path = index_path or config.INDEX_PATH
        ids_path = ids_path or config.INDEX_IDS_PATH
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        np.save(ids_path, self.ids)
        meta = np.array([self.dim], dtype=np.int64)
        np.save(index_path + ".meta.npy", meta)
        if _HAVE_FAISS:
            faiss.write_index(self._index, index_path)
        else:
            np.save(index_path + ".matrix.npy", self._matrix)

    @classmethod
    def load(cls, index_path=None, ids_path=None):
        index_path = index_path or config.INDEX_PATH
        ids_path = ids_path or config.INDEX_IDS_PATH
        dim = int(np.load(index_path + ".meta.npy")[0])
        obj = cls(dim)
        obj.ids = np.load(ids_path)
        if _HAVE_FAISS:
            obj._index = faiss.read_index(index_path)
        else:
            obj._matrix = np.load(index_path + ".matrix.npy")
        return obj
