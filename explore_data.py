"""
Data acquisition & exploration outputs (Task 2).

Produces: sample image grid, class distribution chart, similar vs dissimilar
pairs, intent distribution, vocabulary/token analysis, entity counts, and an
audio waveform + MFCC figure.
"""
import json, os, re, collections
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import librosa, librosa.display
import spacy

ROOT = os.path.join(os.path.dirname(__file__), "..")
IMG = os.path.join(ROOT, "data", "images"); FIG = os.path.join(ROOT, "outputs", "figures")
os.makedirs(FIG, exist_ok=True)

ACCENT = "#3D6BD6"; INK = "#33373D"; MUT = "#6B7280"; GRID = "#E5E7EB"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK,
                     "text.color": INK, "xtick.color": MUT, "ytick.color": MUT,
                     "figure.facecolor": "white"})


def style(ax):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="y" if not ax.get_ygridlines() else "y", color=GRID, lw=.8)
    ax.set_axisbelow(True)


# ---- 1. image class distribution -------------------------------------------
ann = json.load(open(os.path.join(IMG, "annotations.json")))
counts = collections.Counter(a["category"] for a in ann)
cats = sorted(counts, key=counts.get, reverse=True)
fig, ax = plt.subplots(figsize=(7, 3.2))
bars = ax.bar(cats, [counts[c] for c in cats], color=ACCENT, width=.62)
for b, c in zip(bars, cats):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + .4, str(counts[c]),
            ha="center", fontsize=9, color=INK)
ax.set_ylabel("images"); ax.set_title(f"Image dataset – class distribution (n={sum(counts.values())})", loc="left", fontsize=11)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color=GRID, lw=.8); ax.set_axisbelow(True)
plt.xticks(rotation=30, ha="right"); plt.tight_layout()
plt.savefig(os.path.join(FIG, "class_distribution.png"), dpi=160); plt.close()

# ---- 2. similar vs dissimilar examples -------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(7, 5))
pairs = [("gate/gate_000.jpg", "gate/gate_004.jpg", "Similar: two 'gate' signs (same class, different style)"),
         ("information/information_000.jpg", "restaurant/restaurant_001.jpg",
          "Dissimilar: 'information' vs 'restaurant'")]
for r, (f1, f2, title) in enumerate(pairs):
    for c, f in enumerate((f1, f2)):
        axes[r, c].imshow(Image.open(os.path.join(IMG, f))); axes[r, c].axis("off")
    axes[r, 0].set_title(title, loc="left", fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "similar_dissimilar.png"), dpi=160); plt.close()

# ---- 3. sample grid ---------------------------------------------------------
files = ["gate/gate_001.jpg", "baggage_claim/baggage_claim_000.jpg", "check_in/check_in_002.jpg",
         "security/security_000.jpg", "restroom/restroom_001.jpg", "lounge/lounge_000.jpg",
         "transport/transport_002.jpg", "information/information_001.jpg", "restaurant/restaurant_000.jpg"]
fig, axes = plt.subplots(3, 3, figsize=(7.5, 5.6))
for ax, f in zip(axes.flat, files):
    ax.imshow(Image.open(os.path.join(IMG, f))); ax.axis("off")
    ax.set_title(f.split("/")[0], fontsize=9, color=MUT)
plt.suptitle("Representative samples per category", x=.02, ha="left", fontsize=11)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "sample_grid.png"), dpi=160); plt.close()

# ---- 4. text: intent distribution, tokens, entities ------------------------
df = pd.read_csv(os.path.join(ROOT, "data", "text", "queries.csv"))
ic = df["intent"].value_counts()
fig, ax = plt.subplots(figsize=(7, 3.2))
bars = ax.barh(ic.index[::-1], ic.values[::-1], color=ACCENT, height=.62)
for b, v in zip(bars, ic.values[::-1]):
    ax.text(v + .1, b.get_y() + b.get_height() / 2, str(v), va="center", fontsize=9, color=INK)
ax.set_title("Text dataset – queries per intent (n=%d)" % len(df), loc="left", fontsize=11)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", color=GRID, lw=.8); ax.set_axisbelow(True)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "intent_distribution.png"), dpi=160); plt.close()

nlp = spacy.load("en_core_web_sm")
docs = list(nlp.pipe(df["text"].str.lower()))
toks = [t.text for d in docs for t in d if t.is_alpha and not t.is_stop]
vocab = collections.Counter(toks)
lens = [sum(1 for t in d if not t.is_punct) for d in docs]
stats = {
    "queries": len(df), "intents": df["intent"].nunique(),
    "vocab_size_content": len(vocab),
    "mean_query_len_tokens": round(float(np.mean(lens)), 2),
    "min_len": int(min(lens)), "max_len": int(max(lens)),
    "top15_content_words": vocab.most_common(15),
    "entity_annotations": collections.Counter(
        e.split("=")[0] for es in df["entities"].dropna() for e in str(es).split(";") if "=" in e
    ).most_common(),
}
json.dump(stats, open(os.path.join(ROOT, "outputs", "eval", "text_exploration.json"), "w"), indent=2)
print(json.dumps(stats, indent=2))

# ---- 5. audio example: waveform + MFCC -------------------------------------
y, sr = librosa.load(os.path.join(ROOT, "data", "audio", "q000.wav"), sr=16000)
yn, _ = librosa.load(os.path.join(ROOT, "data", "audio", "q000_noisy.wav"), sr=16000)
fig, axes = plt.subplots(2, 2, figsize=(8, 4.4))
for i, (sig, name) in enumerate([(y, "clean"), (yn, "noisy")]):
    axes[0, i].plot(np.arange(len(sig)) / sr, sig, lw=.4, color=ACCENT)
    axes[0, i].set_title(f'Waveform – "Where is gate B12?" ({name})', fontsize=9, loc="left")
    axes[0, i].set_xlabel("s"); axes[0, i].spines[["top", "right"]].set_visible(False)
    m = librosa.feature.mfcc(y=sig, sr=sr, n_mfcc=13)
    im = axes[1, i].imshow(m, aspect="auto", origin="lower", cmap="Blues")
    axes[1, i].set_title(f"MFCC (13 coeff, {name})", fontsize=9, loc="left")
    axes[1, i].set_xlabel("frame"); axes[1, i].set_ylabel("coeff")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "audio_example.png"), dpi=160); plt.close()
print("figures written to", os.path.abspath(FIG))
