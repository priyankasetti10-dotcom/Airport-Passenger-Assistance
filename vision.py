"""
Vision model (Task 4): compact CNN classifier for airport sign categories,
whose penultimate-layer embeddings are indexed with FAISS for image->knowledge
base retrieval (embedding + similarity search design, in the spirit of the
CLIP+FAISS approach but with a trainable backbone so the full training
pipeline can be demonstrated end-to-end on the curated dataset).
"""
import os, json
import numpy as np
import torch
import torch.nn as nn
import faiss
from PIL import Image

from preprocessing import make_loaders, eval_tfms, load_image_splits, IMG_DIR

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH = os.path.join(ROOT, "outputs", "sign_cnn.pt")
INDEX_PATH = os.path.join(ROOT, "outputs", "sign_index.faiss")
EMB_DIM = 128


class SignCNN(nn.Module):
    """4-block ConvNet: ~420k params, suitable for a small curated dataset."""

    def __init__(self, n_classes, emb_dim=EMB_DIM):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                                 nn.ReLU(), nn.MaxPool2d(2))
        self.features = nn.Sequential(block(3, 32), block(32, 64),
                                      block(64, 96), block(96, 128),
                                      nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.embed = nn.Sequential(nn.Linear(128, emb_dim), nn.ReLU())
        self.head = nn.Linear(emb_dim, n_classes)

    def forward(self, x, return_embedding=False):
        e = self.embed(self.features(x))
        return e if return_embedding else self.head(e)


def train_model(epochs=60, lr=2e-3, seed=42):
    torch.manual_seed(seed)
    tr, va, te, classes = make_loaders()
    model = SignCNN(len(classes))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.CrossEntropyLoss()
    history = []
    for ep in range(epochs):
        model.train(); tl, tn, tc = 0, 0, 0
        for x, y in tr:
            opt.zero_grad(); out = model(x); loss = lossf(out, y)
            loss.backward(); opt.step()
            tl += loss.item() * len(y); tn += len(y)
            tc += (out.argmax(1) == y).sum().item()
        model.eval(); vl, vn, vc = 0, 0, 0
        with torch.no_grad():
            for x, y in va:
                out = model(x); vl += lossf(out, y).item() * len(y)
                vn += len(y); vc += (out.argmax(1) == y).sum().item()
        sched.step()
        history.append({"epoch": ep + 1, "train_loss": tl / tn, "train_acc": tc / tn,
                        "val_loss": vl / vn, "val_acc": vc / vn})
        print(f"ep {ep+1:02d} train_loss {tl/tn:.3f} acc {tc/tn:.3f} | val_loss {vl/vn:.3f} acc {vc/vn:.3f}")
    torch.save({"state": model.state_dict(), "classes": classes}, MODEL_PATH)
    json.dump(history, open(os.path.join(ROOT, "outputs", "eval", "train_history.json"), "w"), indent=1)
    return model, classes, history


def load_model():
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model = SignCNN(len(ckpt["classes"]))
    model.load_state_dict(ckpt["state"]); model.eval()
    return model, ckpt["classes"]


@torch.no_grad()
def embed_image(model, pil_img):
    x = eval_tfms(pil_img.convert("RGB")).unsqueeze(0)
    e = model(x, return_embedding=True).numpy().astype("float32")
    faiss.normalize_L2(e)
    return e


def build_index(model=None):
    """Index TRAIN+VAL image embeddings in FAISS (cosine via inner product)."""
    if model is None:
        model, classes = load_model()
    tr, va, te, classes = load_image_splits()
    gallery = tr + va                       # test images stay out of the index
    embs, meta = [], []
    for r in gallery:
        img = Image.open(os.path.join(IMG_DIR, r["file"]))
        embs.append(embed_image(model, img)[0])
        meta.append(r)
    embs = np.vstack(embs).astype("float32")
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    faiss.write_index(index, INDEX_PATH)
    json.dump(meta, open(INDEX_PATH + ".meta.json", "w"))
    print(f"FAISS index: {index.ntotal} vectors, dim {embs.shape[1]}")
    return index, meta


def load_index():
    index = faiss.read_index(INDEX_PATH)
    meta = json.load(open(INDEX_PATH + ".meta.json"))
    return index, meta


def classify_image(pil_img, model=None, classes=None, index=None, meta=None, k=5):
    """Classifier softmax + FAISS neighbour vote -> category + confidence."""
    if model is None:
        model, classes = load_model()
    if index is None:
        index, meta = load_index()
    x = eval_tfms(pil_img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        e = model(x, return_embedding=True)
        logits = model.head(e)
        probs = torch.softmax(logits, 1)[0].numpy()
    q = e.numpy().astype("float32"); faiss.normalize_L2(q)
    sims, idx = index.search(q, k)
    neighbours = [{"file": meta[i]["file"], "category": meta[i]["category"],
                   "similarity": float(s)} for i, s in zip(idx[0], sims[0])]
    order = np.argsort(probs)[::-1]
    top = [{"category": classes[i], "prob": float(probs[i])} for i in order[:3]]
    return {"top3": top, "category": top[0]["category"],
            "confidence": top[0]["prob"], "neighbours": neighbours}


if __name__ == "__main__":
    model, classes, hist = train_model()
    build_index(model)
