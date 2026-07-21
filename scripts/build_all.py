"""One-shot project setup: generate samples, seed the catalog, train the
classifier, build the similarity index and train the sales model.

Usage:
    python scripts/build_all.py [--per-class 24] [--estimator rf]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import generate_samples          # noqa: E402
import seed_db                   # noqa: E402
import train_classifier          # noqa: E402
import build_index               # noqa: E402
import train_sales               # noqa: E402


def step(title):
    print("\n" + "=" * 60)
    print("  " + title)
    print("=" * 60)


def main(per_class, estimator):
    step("1/5  Generating sample swatches")
    generate_samples.build(per_class)

    step("2/5  Seeding SQLite catalog + 2018 sales history")
    seed_db.seed()

    step("3/5  Training pattern classifier")
    train_classifier.main(estimator)

    step("4/5  Building similarity index")
    build_index.main()

    step("5/5  Training sales-prediction model")
    train_sales.main()

    print("\nAll artifacts built. Launch the app with:  npm start")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=24)
    ap.add_argument("--estimator", choices=["rf", "svm"], default="rf")
    args = ap.parse_args()
    main(args.per_class, args.estimator)
