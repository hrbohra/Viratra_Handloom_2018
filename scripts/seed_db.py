"""Populate the SQLite catalog from the sample swatches and generate a year of
synthetic 2018 sales history.

Each sample becomes a saree record with a plausible SKU, region, fabric and
price plus its extracted dominant colours. Sales are drawn from a demand model
that depends on pattern popularity and price, so the Layer-9 sales regressor has
a real signal to learn.

Usage:
    python scripts/seed_db.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viratra import color, config, preprocess          # noqa: E402
from viratra.db import Database                          # noqa: E402

# Relative pattern popularity (drives synthetic demand).
POPULARITY = {
    "floral": 1.35, "paisley": 1.15, "temple": 1.25, "checks": 0.9,
    "stripes": 0.8, "geometric": 1.0, "butta": 1.1, "abstract": 0.7,
}
FABRIC_BASE = {
    "Silk": 8500, "Cotton": 1800, "Cotton-Silk": 3800,
    "Tussar": 5200, "Georgette": 2600, "Brocade": 9500,
}
MONTHS = ["2018-{:02d}".format(m) for m in range(1, 13)]


def _sku(key, i):
    return "VH-{}-{:03d}".format(key[:3].upper(), i)


def seed():
    config.ensure_dirs()
    if os.path.exists(config.DB_PATH):
        os.remove(config.DB_PATH)

    db = Database()
    db.init_schema()
    tax = config.load_taxonomy()
    regions, fabrics = tax["regions"], tax["fabrics"]

    rng = np.random.RandomState(2018)
    n_sarees = 0

    for p in tax["patterns"]:
        key = p["key"]
        sample_dir = os.path.join(config.SAMPLES_DIR, key)
        if not os.path.isdir(sample_dir):
            continue
        files = sorted(f for f in os.listdir(sample_dir) if f.endswith(".png"))
        for i, fname in enumerate(files):
            path = os.path.join(sample_dir, fname)
            region = regions[rng.randint(len(regions))]
            fabric = fabrics[rng.randint(len(fabrics))]
            price = round(FABRIC_BASE[fabric] * (0.8 + 0.5 * rng.rand()), -1)

            prep = preprocess.prepare(path)
            colors = color.dominant_colors(prep["bgr"], k=5)

            saree_id = db.insert_saree(
                sku=_sku(key, i),
                title="{} {} saree".format(region, p["label"]),
                region=region, fabric=fabric, price=price,
                image_path=os.path.relpath(path, config.ROOT).replace("\\", "/"),
                pattern=key, dominant_colors=colors,
            )
            n_sarees += 1

            # --- synthetic monthly sales -----------------------------------
            base = POPULARITY[key] * (12000.0 / max(price, 500.0))
            for month in MONTHS:
                season = 1.0 + 0.35 * np.sin(int(month[-2:]) / 12.0 * 2 * np.pi)
                units = int(max(0, rng.poisson(max(0.2, base * season))))
                if units == 0:
                    continue
                db.insert_sale(saree_id, units, round(units * price, 2),
                               month + "-15")

    print("Seeded {} sarees across {} patterns".format(n_sarees, len(tax["patterns"])))
    print("Sales rows: {}".format(
        db.conn.execute("SELECT COUNT(*) FROM sales").fetchone()[0]))
    db.close()


if __name__ == "__main__":
    seed()
