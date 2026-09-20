"""
Preprocessing pipelines (Task 3): image, audio, and text.

Typed text queries and transcribed voice queries converge on the SAME text
pipeline (clean_text -> tokenise -> vectorise), so the NLP stack is
modality-agnostic downstream of speech-to-text.
"""
import os, re, json, io
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import librosa
import soundfile as sf
import speech_recognition as sr
import spacy

ROOT = os.path.join(os.path.dirname(__file__), "..")
IMG_DIR = os.path.join(ROOT, "data", "images")

# --------------------------------------------------------------------------
# IMAGE PIPELINE: load -> resize -> (augment) -> tensor -> normalise
# --------------------------------------------------------------------------
IMG_MEAN = [0.485, 0.456, 0.406]   # ImageNet statistics, standard practice
IMG_STD = [0.229, 0.224, 0.225]

train_tfms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomRotation(5, fill=(200, 200, 200)),
    transforms.RandomResizedCrop(112, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
    transforms.ColorJitter(brightness=0.2, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(IMG_MEAN, IMG_STD),
])
eval_tfms = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(IMG_MEAN, IMG_STD),
])


class SignDataset(Dataset):
    """Airport sign images + integer category labels."""

    def __init__(self, records, classes, tfms):
        self.records, self.classes, self.tfms = records, classes, tfms
        self.cls2idx = {c: i for i, c in enumerate(classes)}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        img = Image.open(os.path.join(IMG_DIR, r["file"])).convert("RGB")
        return self.tfms(img), self.cls2idx[r["category"]]


def load_image_splits(seed=42, val_frac=0.2, test_frac=0.2):
    """Stratified train/val/test split of the annotated image set."""
    ann = json.load(open(os.path.join(IMG_DIR, "annotations.json")))
    classes = sorted({a["category"] for a in ann})
    rng = np.random.RandomState(seed)
    tr, va, te = [], [], []
    for c in classes:
        recs = [a for a in ann if a["category"] == c]
        rng.shuffle(recs)
        n = len(recs); nv, nt = int(n * val_frac), int(n * test_frac)
        va += recs[:nv]; te += recs[nv:nv + nt]; tr += recs[nv + nt:]
    return tr, va, te, classes


def make_loaders(batch_size=16, seed=42):
    tr, va, te, classes = load_image_splits(seed)
    return (DataLoader(SignDataset(tr, classes, train_tfms), batch_size, shuffle=True),
            DataLoader(SignDataset(va, classes, eval_tfms), batch_size),
            DataLoader(SignDataset(te, classes, eval_tfms), batch_size),
            classes)


# --------------------------------------------------------------------------
# AUDIO PIPELINE: load -> resample 16 kHz mono -> trim silence -> STT / MFCC
# --------------------------------------------------------------------------
def load_audio(path, sr_target=16000, trim_db=25):
    y, _ = librosa.load(path, sr=sr_target, mono=True)
    y, _ = librosa.effects.trim(y, top_db=trim_db)
    return y, sr_target


def audio_mfcc(y, sr_=16000, n_mfcc=13):
    return librosa.feature.mfcc(y=y, sr=sr_, n_mfcc=n_mfcc)


def transcribe(path):
    """Offline speech-to-text: domain-adapted PocketSphinx (see speech.py).

    Returns the transcript text only; use speech.transcribe for (text, mode).
    """
    from speech import transcribe as _tx
    return _tx(path)[0]


# --------------------------------------------------------------------------
# TEXT PIPELINE: clean -> tokenise -> entities -> vectorise
# Shared by typed queries AND speech transcripts.
# --------------------------------------------------------------------------
_nlp = spacy.load("en_core_web_sm")

GATE_RE = re.compile(r"\bgate[s]?\s*([a-d])\s*[- ]?\s*(\d{1,2})(?:\s*(?:to|-)\s*[a-d]?\s*(\d{1,2}))?", re.I)
TERM_RE = re.compile(r"\bterminal\s*(\d)\b", re.I)
FLIGHT_RE = re.compile(r"\b([a-z]{2})\s?(\d{2,4})\b(?=.*flight)|\bflight\s+([a-z]{2})\s?(\d{2,4})\b", re.I)
DESK_RE = re.compile(r"\bdesk[s]?\s*(\d{1,2})(?:\s*(?:to|-)\s*(\d{1,2}))?", re.I)


def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s'?-]", " ", text)
    return re.sub(r"\s+", " ", text)


def tokenise(text, remove_stop=True):
    doc = _nlp(clean_text(text))
    return [t.lemma_ for t in doc if t.is_alpha and not (remove_stop and t.is_stop)]


def extract_entities(text):
    """Hybrid rule-based + spaCy NER entity extraction."""
    ents = {}
    t = text.lower()
    m = GATE_RE.search(t)
    if m:
        g = f"{m.group(1).upper()}{m.group(2)}"
        if m.group(3):
            g += f"-{m.group(1).upper()}{m.group(3)}"
        ents["gate"] = g
    m = TERM_RE.search(t)
    if m:
        ents["terminal"] = m.group(1)
    m = FLIGHT_RE.search(text)
    if m:
        grp = [g for g in m.groups() if g]
        if len(grp) >= 2:
            ents["flight"] = (grp[0] + grp[1]).upper()
    m = DESK_RE.search(t)
    if m:
        ents["desk"] = m.group(1) + (f"-{m.group(2)}" if m.group(2) else "")
    for kw, label in [("wheelchair", "service"), ("lost and found", "service"),
                      ("fast track", "service"), ("halal", "service"),
                      ("taxi", "service"), ("train", "service"), ("bus", "service"),
                      ("lounge", "service"), ("prayer", "service"), ("first aid", "service"),
                      ("car rental", "service"), ("rent a car", "service")]:
        if kw in t and "service" not in ents:
            ents["service"] = kw
    for e in _nlp(text).ents:                      # spaCy NER for places/times
        if e.label_ in ("GPE", "LOC", "TIME") and e.label_.lower() not in ents:
            ents[e.label_.lower()] = e.text
    return ents


def load_text_dataset():
    df = pd.read_csv(os.path.join(ROOT, "data", "text", "queries.csv"))
    df["clean"] = df["text"].map(clean_text)
    df["tokens"] = df["clean"].map(lambda s: " ".join(tokenise(s, remove_stop=False)))
    return df
