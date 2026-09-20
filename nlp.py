"""
NLP components (Task 4): intent classification, entity extraction, and
knowledge-base retrieval.

- Intent classifier: TF-IDF (word 1-2 grams + char 3-5 grams) -> linear
  Logistic Regression. Compact, fast, fully offline, and strong on short
  queries; DistilBERT fine-tuning is a drop-in upgrade when GPU/network allow.
- KB retrieval: TF-IDF vectors over each record's searchable text, cosine
  similarity, with intent/category priors and entity-based boosts.
"""
import os, json, re
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from preprocessing import clean_text, extract_entities, load_text_dataset

ROOT = os.path.join(os.path.dirname(__file__), "..")
KB_PATH = os.path.join(ROOT, "kb", "airport_kb.json")
CLF_PATH = os.path.join(ROOT, "outputs", "intent_clf.joblib")

# intent -> KB categories that plausibly answer it
INTENT2CATS = {
    "find_gate": ["gate"], "baggage_claim": ["baggage_claim"],
    "check_in": ["check_in"], "security": ["security"],
    "restroom": ["restroom"], "lounge": ["lounge"], "transport": ["transport"],
    "information": ["information"], "restaurant": ["restaurant"],
    "flight_status": ["gate", "information"],
    "special_assistance": ["information"],
}
CONFIDENCE_FLOOR = 0.35          # below this -> "not sure" answer


def load_kb():
    kb = json.load(open(KB_PATH))
    return kb["records"]


def record_text(r):
    return " ".join([r["name"], r["category"].replace("_", " "), r["terminal"],
                     r["floor_zone"], r["description"], r["directions"],
                     r["accessibility"], " ".join(r.get("related", []))]).lower()


def build_intent_classifier(seed=42, save=True):
    df = load_text_dataset()
    Xtr, Xte, ytr, yte = train_test_split(df["clean"], df["intent"],
                                          test_size=0.3, stratify=df["intent"],
                                          random_state=seed)
    feats = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
    ])
    clf = Pipeline([("feats", feats),
                    ("lr", LogisticRegression(max_iter=2000, C=8.0))])
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    report = classification_report(yte, pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(yte, pred, labels=sorted(df["intent"].unique()))
    metrics = {"test_accuracy": float(accuracy_score(yte, pred)),
               "report": report,
               "labels": sorted(df["intent"].unique()),
               "confusion_matrix": cm.tolist(),
               "n_train": len(Xtr), "n_test": len(Xte)}
    if save:
        joblib.dump(clf, CLF_PATH)
        json.dump(metrics, open(os.path.join(ROOT, "outputs", "eval", "intent_metrics.json"), "w"), indent=1)
    return clf, metrics, (Xte.tolist(), yte.tolist(), pred.tolist())


def load_intent_classifier():
    return joblib.load(CLF_PATH)


class KBRetriever:
    """TF-IDF cosine retrieval over KB records with intent + entity boosts."""

    def __init__(self):
        self.records = load_kb()
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.M = self.vec.fit_transform([record_text(r) for r in self.records])

    def search(self, query, intent=None, entities=None, k=3):
        q = self.vec.transform([clean_text(query)])
        sims = (self.M @ q.T).toarray().ravel()
        # normalise to 0..1 (cosine on l2-normalised tf-idf is already 0..1)
        boosts = np.zeros_like(sims)
        cats = INTENT2CATS.get(intent, [])
        for i, r in enumerate(self.records):
            if r["category"] in cats:
                boosts[i] += 0.15
            if entities:
                if "gate" in entities and entities["gate"].split("-")[0].lower() in r["name"].lower():
                    boosts[i] += 0.4
                if "terminal" in entities and f"terminal {entities['terminal']}" in r["terminal"].lower():
                    boosts[i] += 0.1
                if "service" in entities and entities["service"] in record_text(self.records[i]):
                    boosts[i] += 0.2
        score = sims + boosts
        order = np.argsort(score)[::-1][:k]
        return [{"record": self.records[i], "score": float(score[i]),
                 "cosine": float(sims[i])} for i in order]


def understand(text, clf=None):
    """Full text-side NLU: intent + confidence + entities."""
    if clf is None:
        clf = load_intent_classifier()
    clean = clean_text(text)
    probs = clf.predict_proba([clean])[0]
    order = np.argsort(probs)[::-1]
    intents = [{"intent": clf.classes_[i], "prob": float(probs[i])} for i in order[:3]]
    return {"intent": intents[0]["intent"], "confidence": intents[0]["prob"],
            "top3": intents, "entities": extract_entities(text)}


if __name__ == "__main__":
    clf, metrics, _ = build_intent_classifier()
    print("intent test accuracy:", round(metrics["test_accuracy"], 3))
    r = KBRetriever()
    for q in ["Where is gate B12?", "I lost my phone, where is lost and found?",
              "Is there a lounge near terminal 2?"]:
        nlu = understand(q, clf)
        hit = r.search(q, nlu["intent"], nlu["entities"], k=1)[0]
        print(f"{q!r:55s} -> {nlu['intent']:18s} {hit['record']['name']} ({hit['score']:.2f})")
