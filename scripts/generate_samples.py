"""Synthesise a labelled sample catalog of saree swatches, one folder per
pattern class. These procedural swatches give the engine a reproducible,
licence-clean dataset to train the classifier and build the similarity index,
without shipping any third-party photographs.

Real shop photos can be dropped into data/samples/real/ and ingested with the
CLI at any time; the synthetic set simply guarantees the demo runs end-to-end.

Usage:
    python scripts/generate_samples.py --per-class 24
"""

import argparse
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viratra import config  # noqa: E402

SIZE = 400

# Saree-like palettes: (body, motif, accent-gold). BGR-free, plain RGB.
PALETTES = [
    ((139, 0, 34), (255, 214, 89), (212, 175, 55)),    # deep red + gold
    ((0, 76, 61), (240, 230, 200), (212, 175, 55)),    # bottle green + cream
    ((26, 42, 108), (233, 196, 106), (212, 175, 55)),  # indigo + mustard
    ((120, 30, 90), (245, 222, 179), (212, 175, 55)),  # magenta + wheat
    ((153, 51, 0), (255, 236, 179), (212, 175, 55)),   # rust + pale gold
    ((45, 45, 45), (224, 176, 96), (212, 175, 55)),    # charcoal + antique gold
    ((0, 90, 120), (250, 240, 215), (212, 175, 55)),   # teal + ivory
    ((90, 20, 20), (238, 214, 175), (212, 175, 55)),   # maroon + sand
]


def _rng(pattern_id, i):
    return np.random.RandomState(pattern_id * 1000 + i)


def _palette(rng):
    body, motif, gold = PALETTES[rng.randint(len(PALETTES))]
    jitter = lambda c: tuple(int(np.clip(v + rng.randint(-15, 16), 0, 255)) for v in c)
    return jitter(body), jitter(motif), gold


def _base(body):
    img = Image.new("RGB", (SIZE, SIZE), body)
    return img, ImageDraw.Draw(img)


def _border(draw, gold, w=26):
    for off, width in ((0, w), (w + 6, 4)):
        draw.rectangle([off, off, SIZE - 1 - off, SIZE - 1 - off], outline=gold, width=width)


# ---- per-class generators ------------------------------------------------
def gen_floral(draw, rng, motif, gold):
    for _ in range(rng.randint(10, 16)):
        cx, cy = rng.randint(40, SIZE - 40, size=2)
        r = rng.randint(14, 26)
        for k in range(6):
            a = k * math.pi / 3
            x, y = cx + r * math.cos(a), cy + r * math.sin(a)
            draw.ellipse([x - r * 0.6, y - r * 0.6, x + r * 0.6, y + r * 0.6],
                         fill=motif, outline=gold)
        draw.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=gold)


def _teardrop(cx, cy, s, rot, bend=0.6):
    """Return polygon points of a mango/kairi (paisley) teardrop."""
    pts = []
    ca, sa = math.cos(rot), math.sin(rot)
    for th in np.linspace(0, 2 * math.pi, 64):
        x = math.cos(th)
        y = math.sin(th) * (math.sin(th / 2) ** 3)
        x += bend * (y ** 2)                       # hook the tip into a curl
        px, py = s * x, s * 2.2 * y
        pts.append((cx + px * ca - py * sa, cy + px * sa + py * ca))
    return pts


def gen_paisley(draw, rng, motif, gold):
    for _ in range(rng.randint(7, 11)):
        cx, cy = rng.randint(60, SIZE - 60, size=2)
        s = rng.randint(26, 40)
        rot = rng.rand() * 2 * math.pi
        draw.polygon(_teardrop(cx, cy, s, rot), fill=motif, outline=gold)
        # inner filigree teardrop
        inner = _teardrop(cx, cy, s * 0.55, rot)
        draw.line(inner + [inner[0]], fill=gold, width=3)


def gen_temple(draw, rng, motif, gold):
    step = rng.randint(30, 42)
    for band_y in (40, SIZE - 40 - step):
        x = 30
        while x < SIZE - 30:
            up = band_y == 40
            tip = (band_y + step) if up else band_y
            base = band_y if up else (band_y + step)
            draw.polygon([(x, base), (x + step, base), (x + step / 2, tip)],
                         fill=motif, outline=gold)
            x += step
    draw.rectangle([30, SIZE // 2 - 6, SIZE - 30, SIZE // 2 + 6], fill=gold)


def gen_checks(draw, rng, motif, gold):
    step = rng.randint(28, 40)
    for x in range(30, SIZE - 30, step):
        draw.line([(x, 30), (x, SIZE - 30)], fill=motif, width=6)
    for y in range(30, SIZE - 30, step):
        draw.line([(30, y), (SIZE - 30, y)], fill=motif, width=6)
    for x in range(30, SIZE - 30, step * 2):
        for y in range(30, SIZE - 30, step * 2):
            draw.rectangle([x, y, x + step, y + step], outline=gold)


def gen_stripes(draw, rng, motif, gold):
    step = rng.randint(18, 28)
    for i, x in enumerate(range(30, SIZE - 30, step)):
        col = gold if i % 3 == 0 else motif
        draw.rectangle([x, 30, x + step // 2, SIZE - 30], fill=col)


def gen_geometric(draw, rng, motif, gold):
    step = rng.randint(34, 46)
    for y in range(30, SIZE - 30, step):
        for x in range(30, SIZE - 30, step):
            cx, cy = x + step / 2, y + step / 2
            d = step * 0.45
            draw.polygon([(cx, cy - d), (cx + d, cy), (cx, cy + d), (cx - d, cy)],
                         fill=motif if (x + y) % (step * 2) else gold, outline=gold)


def gen_butta(draw, rng, motif, gold):
    step = rng.randint(44, 60)
    for y in range(40, SIZE - 30, step):
        for x in range(40, SIZE - 30, step):
            jx, jy = rng.randint(-5, 6, size=2)
            cx, cy = x + jx, y + jy
            draw.ellipse([cx - 8, cy - 8, cx + 8, cy + 8], fill=motif, outline=gold)
            draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=gold)


def gen_abstract(draw, rng, motif, gold):
    for _ in range(rng.randint(14, 22)):
        cx, cy = rng.randint(40, SIZE - 40, size=2)
        w, h = rng.randint(20, 60, size=2)
        col = motif if rng.rand() > 0.4 else gold
        shape = rng.randint(3)
        box = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
        if shape == 0:
            draw.ellipse(box, fill=col)
        elif shape == 1:
            draw.arc(box, rng.randint(0, 180), rng.randint(180, 360), fill=col, width=6)
        else:
            draw.line([cx - w, cy, cx + w, cy + h], fill=col, width=8)


GENERATORS = {
    "floral": gen_floral,
    "paisley": gen_paisley,
    "temple": gen_temple,
    "checks": gen_checks,
    "stripes": gen_stripes,
    "geometric": gen_geometric,
    "butta": gen_butta,
    "abstract": gen_abstract,
}


def build(per_class):
    config.ensure_dirs()
    tax = config.load_taxonomy()
    total = 0
    for p in tax["patterns"]:
        key = p["key"]
        out_dir = os.path.join(config.SAMPLES_DIR, key)
        os.makedirs(out_dir, exist_ok=True)
        for i in range(per_class):
            rng = _rng(p["id"], i)
            body, motif, gold = _palette(rng)
            img, draw = _base(body)
            GENERATORS[key](draw, rng, motif, gold)
            _border(draw, gold)
            path = os.path.join(out_dir, "{}_{:02d}.png".format(key, i))
            img.save(path)
            total += 1
        print("  {:10s} -> {} swatches".format(key, per_class))
    print("Generated {} sample swatches under {}".format(total, config.SAMPLES_DIR))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate synthetic saree swatches")
    ap.add_argument("--per-class", type=int, default=24)
    args = ap.parse_args()
    build(args.per_class)
