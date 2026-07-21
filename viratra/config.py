"""Central configuration and taxonomy loading.

All paths are resolved relative to the project root so the engine behaves the
same whether it is invoked from the CLI, the test-suite or the Electron main
process.
"""

import json
import os

# Project root = one level above this package.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(ROOT, "data")
SAMPLES_DIR = os.path.join(DATA_DIR, "samples")
MODELS_DIR = os.path.join(DATA_DIR, "models")
DB_PATH = os.path.join(DATA_DIR, "viratra.db")

CLASSIFIER_PATH = os.path.join(MODELS_DIR, "pattern_clf.pkl")
INDEX_PATH = os.path.join(MODELS_DIR, "fingerprints.index")
INDEX_IDS_PATH = os.path.join(MODELS_DIR, "fingerprints_ids.npy")
SALES_MODEL_PATH = os.path.join(MODELS_DIR, "sales_reg.pkl")

CONFIG_DIR = os.path.join(ROOT, "config")
TAXONOMY_PATH = os.path.join(CONFIG_DIR, "taxonomy.json")

# Canonical working size for every image the engine sees. Keeping this fixed
# makes histograms, LBP and Gabor energies comparable across the catalog.
IMAGE_SIZE = (256, 256)


def load_taxonomy():
    """Return the parsed taxonomy dict from config/taxonomy.json."""
    with open(TAXONOMY_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def pattern_keys():
    """Ordered list of pattern keys, indexed by class id."""
    tax = load_taxonomy()
    ordered = sorted(tax["patterns"], key=lambda p: p["id"])
    return [p["key"] for p in ordered]


def pattern_labels():
    """Map pattern key -> human label."""
    tax = load_taxonomy()
    return {p["key"]: p["label"] for p in tax["patterns"]}


def ensure_dirs():
    """Create the data directories if they do not yet exist."""
    for d in (DATA_DIR, SAMPLES_DIR, MODELS_DIR):
        os.makedirs(d, exist_ok=True)
