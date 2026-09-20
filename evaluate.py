"""
Training & Evaluation (Task 5): all four pipelines, with tables and figures.

Outputs (outputs/eval/*.json, outputs/figures/*.png):
- vision_metrics.json  : top-1/top-3 accuracy, FAISS retrieval accuracy,
                         similarity stats, stress-test results
- speech_metrics.json  : WER by condition and decoding mode, transcripts,
                         end-to-end voice->intent accuracy
- intent_metrics.json  : held-out P/R/F1 (written by nlp.py) + 5-fold CV here
- retrieval_metrics.json: query->KB record top-1/top-3 accuracy
- fusion_scenarios.json : structured multimodal scenario tests
- figures: training curves, confusion matrix, aug examples, correct/incorrect
"""
import os, json, random
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
import jiwer

from preprocessing import (load_image_splits, eval_tfms, train_tfms, IMG_DIR,
                           clean_text, extract_entities)
from vision import load_model, load_index, classify_image
from nlp import (build_intent_classifier, load_intent_classifier, KBRetriever,
                 understand)
from speech import SphinxTranscriber, spoken_form, written_form

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIG = os.path.join(ROOT, "outputs", "figures")
EVAL = os.path.join(ROOT, "outputs", "eval")
os.makedirs(FIG, exist_ok=True); os.makedirs(EVAL, exist_ok=True)
ACCENT = "#3D6BD6"; INK = "#33373D"; MUT = "#6B7280"; GRID = "#E5E7EB"
random.seed(3); np.random.seed(3)


# ===================== 1. VISION =====================
def degrade(img):
    """Stress transform: heavy blur / darkness / occlusion / downscale."""
    kind = random.choice(["blur", "dark", "occlude", "lowres"])
    if kind == "blur":
        img = img.filter(ImageFilter.GaussianBlur(4.5))
    elif kind == "dark":
        img = ImageEnhance.Brightness(img).enhance(0.28)
    elif kind == "occlude":
        img = img.copy(); d = ImageDraw.Draw(img)
        w, h = img.size
        for _ in range(3):
            x, y = random.randint(0, w - 60), random.randint(0, h - 60)
            d.rectangle([x, y, x + random.randint(50, 110), y + random.randint(40, 90)],
                        fill=(random.randint(0, 80),) * 3)
    else:
        img = img.resize((64, 48)).resize((320, 240))
    return img, kind


def eval_vision():
    model, classes = load_model()
    index, meta = load_index()
    tr, va, te, _ = load_image_splits()
    top1 = top3 = 0; retr1 = 0; sims = []
    rows = []
    for r in te:
        img = Image.open(os.path.join(IMG_DIR, r["file"]))
        out = classify_image(img, model, classes, index, meta)
        cats3 = [t["category"] for t in out["top3"]]
        ok1 = cats3[0] == r["category"]; ok3 = r["category"] in cats3
        top1 += ok1; top3 += ok3
        nb = out["neighbours"][0]
        retr1 += nb["category"] == r["category"]
        sims.append(nb["similarity"])
        rows.append({"file": r["file"], "true": r["category"], "pred": cats3[0],
                     "conf": round(out["confidence"], 3),
                     "faiss_top1": nb["category"], "faiss_sim": round(nb["similarity"], 3),
                     "correct": bool(ok1)})
    n = len(te)
    # stress set
    stress_rows = []; s_top1 = 0
    stress_files = random.sample(te, 27)
    for r in stress_files:
        img = Image.open(os.path.join(IMG_DIR, r["file"]))
        dimg, kind = degrade(img)
        out = classify_image(dimg, model, classes, index, meta)
        ok = out["category"] == r["category"]; s_top1 += ok
        stress_rows.append({"file": r["file"], "degradation": kind, "true": r["category"],
                            "pred": out["category"], "conf": round(out["confidence"], 3),
                            "correct": bool(ok)})
    metrics = {"n_test": n, "top1_acc": top1 / n, "top3_acc": top3 / n,
               "faiss_retrieval_top1_acc": retr1 / n,
               "mean_top1_similarity": float(np.mean(sims)),
               "stress_n": len(stress_rows), "stress_top1_acc": s_top1 / len(stress_rows),
               "test_rows": rows, "stress_rows": stress_rows}
    json.dump(metrics, open(os.path.join(EVAL, "vision_metrics.json"), "w"), indent=1)

    # figures: training curves
    hist = json.load(open(os.path.join(EVAL, "train_history.json")))
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    for ax, key, title in [(axes[0], "loss", "Cross-entropy loss"),
                           (axes[1], "acc", "Accuracy")]:
        ax.plot([h["epoch"] for h in hist], [h[f"train_{key}"] for h in hist],
                color=ACCENT, lw=2, label="train")
        ax.plot([h["epoch"] for h in hist], [h[f"val_{key}"] for h in hist],
                color="#C4483F", lw=2, label="validation")
        ax.set_title(title, loc="left", fontsize=10); ax.set_xlabel("epoch")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color=GRID, lw=.8); ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "training_curves.png"), dpi=160); plt.close()

    # correct vs incorrect / stress examples
    wrong = [r for r in stress_rows if not r["correct"]][:3]
    right = [r for r in rows if r["correct"]][:3]
    fig, axes = plt.subplots(2, 3, figsize=(8, 4.6))
    for i, r in enumerate(right):
        axes[0, i].imshow(Image.open(os.path.join(IMG_DIR, r["file"]))); axes[0, i].axis("off")
        axes[0, i].set_title(f"OK: {r['pred']} ({r['conf']:.2f})", fontsize=9, color="#256D42")
    for i in range(3):
        if i < len(wrong):
            r = wrong[i]
            img = Image.open(os.path.join(IMG_DIR, r["file"]))
            dimg, _ = degrade(img)
            axes[1, i].imshow(dimg); axes[1, i].axis("off")
            axes[1, i].set_title(f"{r['degradation']}: {r['true']} -> {r['pred']} ({r['conf']:.2f})",
                                 fontsize=8.5, color="#B3271E")
        else:
            axes[1, i].axis("off")
    plt.suptitle("Vision: correct test predictions (top) vs stress-set failures (bottom)",
                 x=.02, ha="left", fontsize=10)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "vision_examples.png"), dpi=160); plt.close()

    # augmentation evidence figure (Task 3)
    raw = Image.open(os.path.join(IMG_DIR, te[0]["file"]))
    fig, axes = plt.subplots(1, 5, figsize=(10, 2.4))
    axes[0].imshow(raw); axes[0].set_title("original", fontsize=9); axes[0].axis("off")
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    for i in range(1, 5):
        t = train_tfms(raw)
        img = (t * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        axes[i].imshow(img); axes[i].set_title(f"augmented #{i}", fontsize=9); axes[i].axis("off")
    plt.suptitle("Image pipeline: resize -> rotate -> crop -> colour jitter -> tensor -> normalise",
                 x=.02, ha="left", fontsize=10)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "augmentation_examples.png"), dpi=160); plt.close()
    return metrics


# ===================== 2. SPEECH =====================
def eval_speech():
    tx = SphinxTranscriber()
    m = pd.read_csv(os.path.join(ROOT, "data", "audio", "manifest.csv"))
    clf = load_intent_classifier()
    rows = []
    for _, r in m.iterrows():
        path = os.path.join(ROOT, "data", "audio", r["file"])
        hyp, mode = tx.transcribe(path)
        raw = tx._condition(path)
        open_hyp = tx._decode(tx.open_dec, raw)
        ref = spoken_form(r["text"])
        wer = jiwer.wer(ref, hyp) if hyp else 1.0
        owner = jiwer.wer(ref, open_hyp) if open_hyp else 1.0
        itn = written_form(hyp) if hyp else ""
        pred_intent = understand(itn, clf)["intent"] if itn else None
        rows.append({"file": r["file"], "condition": r["condition"], "voice": r["voice"],
                     "ref": ref, "hyp": hyp, "itn": itn, "mode": mode, "wer": round(wer, 3),
                     "open_hyp": open_hyp, "open_wer": round(owner, 3),
                     "true_intent": r["intent"], "pred_intent": pred_intent,
                     "intent_ok": bool(pred_intent == r["intent"])})
    df = pd.DataFrame(rows)
    def agg(sub):
        return {"n": len(sub),
                "wer_grammar": round(float(jiwer.wer(list(sub["ref"]), list(sub["hyp"].replace("", " ").fillna(" ")))), 3)
                if all(sub["hyp"] != "") else round(float(np.mean(sub["wer"])), 3),
                "mean_wer": round(float(np.mean(sub["wer"])), 3),
                "mean_open_wer": round(float(np.mean(sub["open_wer"])), 3),
                "empty_rate": round(float((sub["hyp"] == "").mean()), 3),
                "voice_intent_acc": round(float(sub["intent_ok"].mean()), 3)}
    metrics = {"overall": agg(df), "clean": agg(df[df.condition == "clean"]),
               "noisy": agg(df[df.condition == "noisy"]),
               "rows": rows}
    json.dump(metrics, open(os.path.join(EVAL, "speech_metrics.json"), "w"), indent=1)
    return metrics


# ===================== 3. TEXT: INTENT + RETRIEVAL =====================
GOLD_RETRIEVAL = {   # query -> expected KB record id
    "Where is gate B12?": "KB01",
    "Take me to gate B12 please": "KB01",
    "Is gate A3 far from here?": "KB02",
    "Where is the check in desk for international flights?": "KB03",
    "Where are check in desks 30 to 40?": "KB03",
    "How do I get to baggage claim?": "KB04",
    "Where do I collect my luggage after landing?": "KB04",
    "Where is security control?": "KB05",
    "Where is the fast track security lane?": "KB05",
    "Where is the nearest information desk?": "KB06",
    "I lost my phone, where is lost and found?": "KB07",
    "Where do I report a lost bag?": "KB07",
    "Is there a lounge near terminal 2?": "KB08",
    "Does the lounge have showers?": "KB08",
    "Where is the food court?": "KB09",
    "Where can I get food before my flight?": "KB09",
    "Where is the nearest toilet?": "KB10",
    "How do I get the train to the city centre?": "KB11",
    "When does the last train leave the airport?": "KB11",
    "Where is the taxi pickup zone?": "KB12",
    "Where do the buses depart from?": "KB13",
    "Is there a night bus to the city?": "KB13",
    "I need wheelchair assistance, where do I go?": "KB14",
    "Where is the special assistance desk?": "KB14",
    "Where is the quiet zone or prayer room?": "KB15",
    "Where can I rent a car?": "KB17",
    "Is there a first aid point in the terminal?": "KB18",
}


def eval_text():
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.pipeline import Pipeline
    clf, metrics, (Xte, yte, pred) = build_intent_classifier()
    df = pd.read_csv(os.path.join(ROOT, "data", "text", "queries.csv"))
    df["clean"] = df["text"].map(clean_text)
    cv = cross_val_score(clf, df["clean"], df["intent"],
                         cv=StratifiedKFold(5, shuffle=True, random_state=42))
    metrics["cv5_mean"] = float(cv.mean()); metrics["cv5_std"] = float(cv.std())

    r = KBRetriever()
    rrows = []; hit1 = hit3 = 0
    for q, gold in GOLD_RETRIEVAL.items():
        nlu = understand(q, clf)
        hits = r.search(q, nlu["intent"], nlu["entities"], k=3)
        ids = [h["record"]["id"] for h in hits]
        hit1 += ids[0] == gold; hit3 += gold in ids
        rrows.append({"query": q, "gold": gold, "top3": ids,
                      "top1_score": round(hits[0]["score"], 3),
                      "intent": nlu["intent"], "entities": nlu["entities"],
                      "correct": bool(ids[0] == gold)})
    n = len(GOLD_RETRIEVAL)
    retrieval = {"n": n, "top1_acc": hit1 / n, "top3_acc": hit3 / n, "rows": rrows}
    json.dump(metrics, open(os.path.join(EVAL, "intent_metrics.json"), "w"), indent=1)
    json.dump(retrieval, open(os.path.join(EVAL, "retrieval_metrics.json"), "w"), indent=1)

    # confusion matrix figure
    cm = np.array(metrics["confusion_matrix"]); labels = metrics["labels"]
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > cm.max() / 2 else INK)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title("Intent classifier - held-out confusion matrix", loc="left", fontsize=10)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "confusion_matrix.png"), dpi=160); plt.close()
    return metrics, retrieval


# ===================== 4. MULTIMODAL FUSION SCENARIOS =====================
def eval_fusion():
    from fusion import AirportAssistant
    bot = AirportAssistant()
    scen = []

    def add(name, modality, expected, **kw):
        res = bot.respond(**kw)
        got = res.get("record", {}).get("id") if res["ok"] else None
        scen.append({"scenario": name, "modality": modality,
                     "input": {k: (v if isinstance(v, str) else "<image>") for k, v in kw.items() if v is not None},
                     "expected": expected, "got": got,
                     "ok_response": res["ok"], "confidence": res.get("confidence"),
                     "fusion_mode": res["details"].get("fusion_mode"),
                     "transcript": res["details"].get("transcript"),
                     "correct": bool(got == expected) if expected else (not res["ok"]),
                     "message_head": res["message"].split("\n")[0]})

    img_gate = Image.open(os.path.join(IMG_DIR, "gate/gate_002.jpg"))
    img_bag = Image.open(os.path.join(IMG_DIR, "baggage_claim/baggage_claim_003.jpg"))
    img_lounge = Image.open(os.path.join(IMG_DIR, "lounge/lounge_004.jpg"))

    add("S1 text only: gate query", "text", "KB01", text="Where is gate B12?")
    add("S2 voice only: baggage claim", "voice", "KB04",
        audio_path=os.path.join(ROOT, "data", "audio", "q003.wav"))
    add("S3 image only: gate sign photo", "image", None if False else "KB01", image=img_gate)
    add("S4 image + text: baggage sign + 'which level?'", "image+text", "KB04",
        image=img_bag, text="which level is this on?")
    add("S5 voice + image: lounge question + lounge sign photo", "voice+image", "KB08",
        image=img_lounge, audio_path=os.path.join(ROOT, "data", "audio", "q015.wav"))
    add("S6 noisy voice: security question", "voice", "KB05",
        audio_path=os.path.join(ROOT, "data", "audio", "q009_noisy.wav"))
    add("S7 out-of-scope text -> uncertain", "text", None,
        text="What is the weather like in Tokyo tomorrow evening?")
    json.dump(scen, open(os.path.join(EVAL, "fusion_scenarios.json"), "w"), indent=1)
    return scen


if __name__ == "__main__":
    v = eval_vision()
    print(f"VISION  top1 {v['top1_acc']:.3f} top3 {v['top3_acc']:.3f} "
          f"faiss {v['faiss_retrieval_top1_acc']:.3f} stress {v['stress_top1_acc']:.3f}")
    s = eval_speech()
    print(f"SPEECH  wer clean {s['clean']['mean_wer']:.3f} noisy {s['noisy']['mean_wer']:.3f} "
          f"open {s['overall']['mean_open_wer']:.3f} voice-intent {s['overall']['voice_intent_acc']:.3f}")
    t, r = eval_text()
    print(f"TEXT    intent acc {t['test_accuracy']:.3f} cv {t['cv5_mean']:.3f}+/-{t['cv5_std']:.3f} "
          f"| retrieval top1 {r['top1_acc']:.3f} top3 {r['top3_acc']:.3f}")
    f = eval_fusion()
    print("FUSION  " + " ".join(f"{x['scenario'].split()[0]}:{'OK' if x['correct'] else 'X'}" for x in f))
