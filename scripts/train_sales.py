"""Train the Layer-9 sales-prediction model.

Joins each saree's attributes (pattern, fabric, region, price) to its total
units sold and fits a RandomForest regressor, then prints which attributes most
drive demand.

Usage:
    python scripts/train_sales.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viratra import config                # noqa: E402
from viratra.db import Database           # noqa: E402
from viratra.sales import SalesModel      # noqa: E402


def main():
    config.ensure_dirs()
    db = Database()
    sarees = db.all_sarees()
    totals = db.sales_by_saree()
    if not sarees:
        print("No catalog - run scripts/seed_db.py first.")
        return

    units = [totals.get(s["id"], {}).get("units", 0) for s in sarees]
    model = SalesModel().fit(sarees, units)
    model.save()

    print("Trained sales model on {} sarees.".format(len(sarees)))
    print("\nTop demand drivers:")
    for row in model.feature_importance()[:8]:
        print("  {:22s} {:.3f}".format(row["attribute"], row["importance"]))
    print("\nSaved -> {}".format(config.SALES_MODEL_PATH))
    db.close()


if __name__ == "__main__":
    main()
