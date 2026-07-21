"""Build the Layer-8 similarity index.

Computes the deep design-fingerprint for every saree in the catalog and builds
the FAISS index (or the scikit-learn fallback), keyed by catalog id, so the
engine can answer "find sarees that look like this one".

Usage:
    python scripts/build_index.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viratra import config, embeddings, preprocess, search   # noqa: E402
from viratra.db import Database                                # noqa: E402


def main():
    config.ensure_dirs()
    db = Database()
    sarees = db.all_sarees()
    if not sarees:
        print("No sarees in catalog - run scripts/seed_db.py first.")
        return

    print("Embedding {} sarees with backend '{}' ...".format(
        len(sarees), embeddings.backend_name()))

    vectors, ids = [], []
    for s in sarees:
        path = os.path.join(config.ROOT, s["image_path"])
        prep = preprocess.prepare(path)
        vectors.append(embeddings.embed(prep))
        ids.append(s["id"])

    vectors = np.vstack(vectors).astype(np.float32)
    index = search.FingerprintIndex(dim=vectors.shape[1])
    index.add(vectors, ids)
    index.save()

    print("Built {} index: {} vectors x {} dims".format(
        search.backend_name(), vectors.shape[0], vectors.shape[1]))
    print("Saved -> {}".format(config.INDEX_PATH))
    db.close()


if __name__ == "__main__":
    main()
