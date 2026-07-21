"""Viratra command-line interface.

A thin, scriptable front-end over the engine for analysing an image, inspecting
the catalog and printing dashboard stats without launching the desktop app.

    python cli/viratra.py analyze path/to/saree.png
    python cli/viratra.py catalog
    python cli/viratra.py stats
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viratra.pipeline import Engine       # noqa: E402


def _swatch(hexcode):
    """Return an ANSI colour block for a #rrggbb string (best-effort)."""
    h = hexcode.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return "\033[48;2;{};{};{}m   \033[0m".format(r, g, b)


def cmd_analyze(engine, args):
    res = engine.analyze(args.image, k=args.k)
    print("\nImage:   {}".format(res["image"]))
    print("Backend: embed={} search={}".format(
        res["embedding_backend"], res["search_backend"]))

    if res["pattern"]:
        p = res["pattern"]
        print("\nPattern: {}  ({:.1%} confidence)".format(p["label"], p["confidence"]))
        ranked = sorted(p["scores"].items(), key=lambda kv: kv[1], reverse=True)[:3]
        print("  top-3: " + ", ".join("{} {:.0%}".format(k, v) for k, v in ranked))
    else:
        print("\nPattern: (classifier not built - run scripts/train_classifier.py)")

    print("\nDominant colours:")
    for c in res["colors"]:
        print("  {} {}  {:.0%}".format(_swatch(c["hex"]), c["hex"], c["fraction"]))

    print("\nSimilar designs:")
    if not res["similar"]:
        print("  (index not built - run scripts/build_index.py)")
    for s in res["similar"]:
        print("  [{:.3f}] {:<14} {:<22} {}".format(
            s["similarity"], s["sku"], s["title"], s.get("pattern", "")))
    print()


def cmd_catalog(engine, args):
    sarees = engine.db.all_sarees()
    print("Catalog: {} sarees".format(len(sarees)))
    for s in sarees[: args.limit]:
        print("  {:<14} {:<26} {:<10} {:>8.0f}  {}".format(
            s["sku"], s["title"], s["pattern"], s["price"] or 0, s["fabric"]))
    if len(sarees) > args.limit:
        print("  ... {} more".format(len(sarees) - args.limit))


def cmd_stats(engine, args):
    db = engine.db
    print("Sarees: {}".format(db.count_sarees()))
    print("\nSales by pattern:")
    for row in db.sales_by_pattern():
        print("  {:<10} units={:<6} revenue={:>12,.0f}".format(
            row["pattern"], row["units"], row["revenue"]))
    if args.json:
        print(json.dumps(db.sales_timeseries(), indent=2))


def main():
    ap = argparse.ArgumentParser(prog="viratra", description="Saree design intelligence CLI")
    sub = ap.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="analyse an image")
    a.add_argument("image")
    a.add_argument("-k", type=int, default=6)
    a.set_defaults(func=cmd_analyze)

    c = sub.add_parser("catalog", help="list catalog sarees")
    c.add_argument("--limit", type=int, default=20)
    c.set_defaults(func=cmd_catalog)

    s = sub.add_parser("stats", help="print sales/inventory stats")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_stats)

    args = ap.parse_args()
    engine = Engine()
    try:
        args.func(engine, args)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
