"""Architecture diagram of the full multimodal pipeline (for the report)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")
INK = "#26292E"; MUT = "#5B6470"
C_IN = "#E8EEFB"; C_PRE = "#FDF3DC"; C_MOD = "#E4F2E8"; C_FUS = "#F3E8F6"
C_KB = "#FDE8E4"; C_OUT = "#E2F0F7"; EDGE = "#9AA4B2"

fig, ax = plt.subplots(figsize=(11.5, 7.6))
ax.set_xlim(0, 115); ax.set_ylim(0, 76); ax.axis("off")


def box(x, y, w, h, colour, title, body="", fs=9.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.6",
                                fc=colour, ec=EDGE, lw=1.1))
    ax.text(x + w / 2, y + h - 3.1, title, ha="center", fontsize=fs, color=INK, weight="bold")
    if body:
        ax.text(x + w / 2, y + (h - 3.1) / 2 - 0.6, body, ha="center", va="center",
                fontsize=fs - 1.3, color=MUT, linespacing=1.45)


def arrow(x1, y1, x2, y2, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=13, color="#7A828E", lw=1.3,
                                 shrinkA=2, shrinkB=2))


# --- inputs ---
box(3, 64, 32, 9.5, C_IN, "IMAGE INPUT", "photo of airport sign /\nboard (JPG, PNG)")
box(41, 64, 32, 9.5, C_IN, "VOICE INPUT", "spoken passenger query\n(WAV, microphone)")
box(79, 64, 32, 9.5, C_IN, "TEXT INPUT", "typed question or\nfollow-up query")

# --- preprocessing ---
box(3, 48.5, 32, 11.5, C_PRE, "Image preprocessing",
    "load  ·  resize 112×112  ·  augment\n(train)  ·  tensor  ·  normalise")
box(41, 48.5, 32, 11.5, C_PRE, "Audio preprocessing",
    "16 kHz mono · trim silence\npad + dither · conditioning")
box(79, 48.5, 32, 11.5, C_PRE, "Text preprocessing",
    "lower-case · clean · tokenise\n(spaCy) · lemmatise")

# --- models ---
box(3, 31.5, 32, 13, C_MOD, "Vision model",
    "CNN classifier (trained)\n128-d embeddings -> FAISS\nsign category + confidence")
box(41, 31.5, 32, 13, C_MOD, "Speech-to-text",
    "PocketSphinx + domain JSGF\ngrammar & custom dictionary\n+ inverse text normalisation")
box(79, 31.5, 32, 13, C_MOD, "NLU",
    "TF-IDF (word+char) + LogReg\nintent · entity extraction\n(rules + spaCy NER)")

# --- fusion ---
box(30, 16, 55, 10.5, C_FUS, "Multimodal fusion & routing",
    "rule-based routing by available modalities\nweighted score fusion  ·  uncertainty threshold")

# --- KB ---
box(90, 16, 22, 10.5, C_KB, "Airport KB", "JSON · 18 records\nschema: name, category,\nterminal, directions, ...")

# --- output ---
box(30, 2, 55, 9.5, C_OUT, "Response generation & UI (Streamlit)",
    "matched record · directions · hours · accessibility · confidence score · fallback to staff")

# arrows
for x in (19, 57, 95):
    arrow(x, 64, x, 60.5)
    arrow(x, 48.5, x, 45)
arrow(57, 31.5, 95, 44.2, style="-|>")   # STT transcript -> NLU (text pipeline)
ax.text(72, 39.5, "transcript ->\nshared text pipeline", fontsize=8, color=MUT, ha="center")
arrow(19, 31.5, 45, 27)                  # vision -> fusion
arrow(95, 31.5, 70, 27)                  # NLU -> fusion
arrow(90, 21, 85.5, 21, style="<|-|>")   # fusion <-> KB
arrow(57, 16, 57, 12)                    # fusion -> output

plt.tight_layout()
plt.savefig(os.path.join(FIG, "architecture.png"), dpi=170, bbox_inches="tight")
print("architecture diagram saved")
