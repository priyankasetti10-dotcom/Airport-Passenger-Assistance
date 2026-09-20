"""
Synthetic airport signage dataset generator.

Creates airport-style sign images (dark signboard + white pictogram + label text)
across 9 passenger-assistance categories, with realistic nuisance variation:
background clutter, sign colour, viewing angle, brightness, blur and sensor noise.

Self-curated synthetic data avoids copyright and privacy issues (no real
passengers, no real airport branding), as permitted by the assignment brief.
"""
import os, json, math, random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

RNG_SEED = 42
random.seed(RNG_SEED); np.random.seed(RNG_SEED)

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "images")
IMG_SIZE = (320, 240)

FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

CATEGORIES = {
    # name: (label texts, images per class -> deliberate mild imbalance)
    "gate":          (["GATE B12", "GATE A3", "GATES B1-B20", "GATE C7"], 78),
    "baggage_claim": (["BAGGAGE CLAIM", "BAGGAGE RECLAIM", "BELTS 1-8"], 66),
    "check_in":      (["CHECK-IN", "CHECK-IN 30-40", "SELF CHECK-IN"], 60),
    "security":      (["SECURITY", "SECURITY CONTROL", "FAST TRACK"], 60),
    "restroom":      (["TOILETS", "RESTROOMS", "WC"], 54),
    "lounge":        (["LOUNGE", "SKYVIEW LOUNGE", "QUIET ZONE"], 48),
    "transport":     (["TRAINS", "TAXI", "BUSES", "CAR RENTAL"], 72),
    "information":   (["INFORMATION", "HELP POINT", "ASSISTANCE"], 54),
    "restaurant":    (["RESTAURANTS", "FOOD COURT", "CAFE"], 48),
}

SIGN_COLOURS = [(20, 32, 66), (12, 12, 14), (0, 70, 48), (48, 18, 68), (90, 60, 10)]
WALL_COLOURS = [(214, 214, 218), (198, 205, 212), (230, 226, 216), (180, 188, 196)]


def _person(d, x, y, s, fill):
    d.ellipse([x - s * .18, y - s * .55, x + s * .18, y - s * .2], fill=fill)
    d.polygon([(x - s * .3, y + s * .55), (x + s * .3, y + s * .55),
               (x + s * .16, y - s * .12), (x - s * .16, y - s * .12)], fill=fill)


def draw_pictogram(d, cat, cx, cy, s, fill=(255, 255, 255)):
    if cat == "gate":  # airplane
        d.polygon([(cx, cy - s * .5), (cx + s * .1, cy - s * .1), (cx + s * .55, cy + s * .15),
                   (cx + s * .55, cy + s * .28), (cx + s * .08, cy + s * .12), (cx + s * .06, cy + s * .38),
                   (cx + s * .22, cy + s * .5), (cx - s * .22, cy + s * .5), (cx - s * .06, cy + s * .38),
                   (cx - s * .08, cy + s * .12), (cx - s * .55, cy + s * .28), (cx - s * .55, cy + s * .15),
                   (cx - s * .1, cy - s * .1)], fill=fill)
    elif cat == "baggage_claim":  # suitcase on belt
        d.rounded_rectangle([cx - s * .35, cy - s * .25, cx + s * .35, cy + s * .3], radius=s * .06, fill=fill)
        d.rectangle([cx - s * .12, cy - s * .42, cx + s * .12, cy - s * .25], outline=fill, width=max(2, int(s * .06)))
        d.line([cx - s * .55, cy + s * .45, cx + s * .55, cy + s * .45], fill=fill, width=max(2, int(s * .05)))
        for px in (-.4, 0, .4):
            d.ellipse([cx + s * px - s * .05, cy + s * .5, cx + s * px + s * .05, cy + s * .6], fill=fill)
    elif cat == "check_in":  # desk + agent + ticket
        _person(d, cx - s * .25, cy - s * .1, s * .5, fill)
        d.rectangle([cx - s * .55, cy + s * .2, cx + s * .55, cy + s * .32], fill=fill)
        d.rectangle([cx - s * .5, cy + s * .32, cx - s * .38, cy + s * .55], fill=fill)
        d.rectangle([cx + s * .38, cy + s * .32, cx + s * .5, cy + s * .55], fill=fill)
        d.polygon([(cx + s * .15, cy - s * .3), (cx + s * .5, cy - s * .42), (cx + s * .5, cy - s * .18),
                   (cx + s * .15, cy - s * .06)], outline=fill, width=max(2, int(s * .05)))
    elif cat == "security":  # shield + tick
        d.polygon([(cx, cy - s * .5), (cx + s * .42, cy - s * .32), (cx + s * .42, cy + s * .05),
                   (cx, cy + s * .5), (cx - s * .42, cy + s * .05), (cx - s * .42, cy - s * .32)],
                  outline=fill, width=max(3, int(s * .08)))
        d.line([cx - s * .18, cy, cx - s * .04, cy + s * .16], fill=fill, width=max(3, int(s * .08)))
        d.line([cx - s * .04, cy + s * .16, cx + s * .22, cy - s * .18], fill=fill, width=max(3, int(s * .08)))
    elif cat == "restroom":  # two figures + divider
        _person(d, cx - s * .28, cy, s * .8, fill)
        d.line([cx, cy - s * .5, cx, cy + s * .5], fill=fill, width=max(2, int(s * .04)))
        _person(d, cx + s * .28, cy, s * .8, fill)
        d.polygon([(cx + s * .28 - s * .3, cy + s * .44), (cx + s * .28 + s * .3, cy + s * .44),
                   (cx + s * .28 + s * .12, cy - s * .1), (cx + s * .28 - s * .12, cy - s * .1)], fill=fill)
    elif cat == "lounge":  # armchair + cup
        d.rounded_rectangle([cx - s * .45, cy - s * .05, cx + s * .2, cy + s * .35], radius=s * .08, fill=fill)
        d.rectangle([cx - s * .45, cy - s * .35, cx - s * .28, cy + s * .35], fill=fill)
        d.rectangle([cx - s * .45, cy + s * .35, cx - s * .38, cy + s * .5], fill=fill)
        d.rectangle([cx + s * .1, cy + s * .35, cx + s * .18, cy + s * .5], fill=fill)
        d.rounded_rectangle([cx + s * .3, cy - s * .3, cx + s * .55, cy - s * .05], radius=s * .04,
                            outline=fill, width=max(2, int(s * .05)))
        d.arc([cx + s * .5, cy - s * .26, cx + s * .64, cy - s * .1], -90, 90, fill=fill, width=max(2, int(s * .05)))
    elif cat == "transport":  # bus front
        d.rounded_rectangle([cx - s * .4, cy - s * .45, cx + s * .4, cy + s * .35], radius=s * .1, fill=fill)
        d.rectangle([cx - s * .3, cy - s * .32, cx + s * .3, cy - s * .02], fill=(60, 60, 60))
        d.ellipse([cx - s * .32, cy + s * .28, cx - s * .12, cy + s * .48], fill=fill)
        d.ellipse([cx + s * .12, cy + s * .28, cx + s * .32, cy + s * .48], fill=fill)
    elif cat == "information":  # circled i
        d.ellipse([cx - s * .45, cy - s * .45, cx + s * .45, cy + s * .45],
                  outline=fill, width=max(3, int(s * .09)))
        d.ellipse([cx - s * .07, cy - s * .3, cx + s * .07, cy - s * .16], fill=fill)
        d.rectangle([cx - s * .06, cy - s * .06, cx + s * .06, cy + s * .3], fill=fill)
    elif cat == "restaurant":  # fork & knife
        d.rectangle([cx - s * .22, cy - s * .5, cx - s * .16, cy + s * .5], fill=fill)
        for off in (-.32, -.19, -.06):
            d.rectangle([cx + s * off - s * .02, cy - s * .5, cx + s * off + s * .02, cy - s * .15], fill=fill)
        d.rectangle([cx - s * .34, cy - s * .18, cx - s * .04, cy - s * .12], fill=fill)
        d.polygon([(cx + s * .2, cy - s * .5), (cx + s * .32, cy - s * .1), (cx + s * .26, cy + s * .5),
                   (cx + s * .2, cy + s * .5)], fill=fill)


def make_sign(cat, label, rng):
    W, H = 640, 480
    wall = random.choice(WALL_COLOURS)
    img = Image.new("RGB", (W, H), wall)
    d = ImageDraw.Draw(img)
    # background clutter: ceiling strip, floor, random silhouettes
    d.rectangle([0, 0, W, 60], fill=tuple(max(0, c - 40) for c in wall))
    d.rectangle([0, H - 70, W, H], fill=tuple(max(0, c - 25) for c in wall))
    for _ in range(random.randint(0, 4)):
        x = random.randint(0, W); h = random.randint(40, 90)
        d.polygon([(x - 12, H - 70), (x + 12, H - 70), (x + 8, H - 70 - h), (x - 8, H - 70 - h)],
                  fill=tuple(max(0, c - random.randint(30, 70)) for c in wall))
    # signboard
    sc = random.choice(SIGN_COLOURS)
    sw, sh = random.randint(420, 540), random.randint(200, 260)
    sx, sy = (W - sw) // 2 + random.randint(-40, 40), random.randint(80, 140)
    d.rounded_rectangle([sx, sy, sx + sw, sy + sh], radius=18, fill=sc,
                        outline=(220, 220, 220), width=2)
    # pictogram + text
    ps = random.randint(70, 90)
    pcx, pcy = sx + 90, sy + sh // 2
    draw_pictogram(d, cat, pcx, pcy, ps)
    fsize = random.randint(34, 44)
    font = ImageFont.truetype(FONT_BOLD, fsize)
    tx = sx + 170
    while d.textlength(label, font=font) > sw - 190 and fsize > 18:
        fsize -= 2; font = ImageFont.truetype(FONT_BOLD, fsize)
    d.text((tx, pcy - fsize * 0.65), label, font=font, fill=(255, 255, 255))
    # directional arrow sometimes
    if random.random() < 0.6:
        ay = sy + sh - 38; ax = sx + sw - 70
        if random.random() < 0.5:
            d.polygon([(ax, ay), (ax + 34, ay + 12), (ax, ay + 24), (ax + 8, ay + 12)], fill=(255, 210, 0))
        else:
            d.polygon([(ax + 34, ay), (ax, ay + 12), (ax + 34, ay + 24), (ax + 26, ay + 12)], fill=(255, 210, 0))
    # nuisance transforms
    img = img.rotate(random.uniform(-7, 7), resample=Image.BICUBIC,
                     fillcolor=wall, expand=False)
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.55, 1.3))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.15))
    if random.random() < 0.45:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.6, 2.2)))
    arr = np.asarray(img).astype(np.float32)
    arr += np.random.normal(0, random.uniform(2, 9), arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return img.resize(IMG_SIZE, Image.LANCZOS)


def main():
    meta = []
    for cat, (labels, n) in CATEGORIES.items():
        cdir = os.path.join(OUT, cat)
        os.makedirs(cdir, exist_ok=True)
        for i in range(n):
            label = labels[i % len(labels)]
            img = make_sign(cat, label, random)
            fn = f"{cat}_{i:03d}.jpg"
            img.save(os.path.join(cdir, fn), quality=88)
            meta.append({"file": f"{cat}/{fn}", "category": cat, "sign_text": label})
    with open(os.path.join(OUT, "annotations.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"Generated {len(meta)} images across {len(CATEGORIES)} categories -> {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
