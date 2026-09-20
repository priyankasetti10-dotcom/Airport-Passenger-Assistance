"""
Multimodal fusion (Task 4): rule-based routing + weighted score fusion.

Input scenarios handled: image-only, voice-only, text-only, image+text,
image+voice. Voice is first transcribed, then shares the text pipeline.

Fusion strategy
---------------
1. ROUTING: each present modality is processed by its own expert
   (CNN+FAISS for images; STT -> NLU for audio; NLU for text).
2. WEIGHTED SCORE FUSION over knowledge-base records:
       score(r) = w_t * text_score(r) + w_i * image_score(r)
   where image_score maps the predicted sign category onto matching KB
   records weighted by classifier confidence, and weights depend on which
   modalities are present and how confident each expert is.
3. UNCERTAINTY: if the fused best score < threshold, the bot declines to
   guess and refers the passenger to staff (avoids false certainty).
"""
import numpy as np

from nlp import KBRetriever, understand, load_intent_classifier, INTENT2CATS, CONFIDENCE_FLOOR
from vision import load_model, load_index, classify_image
from preprocessing import transcribe

W_TEXT, W_IMAGE = 0.65, 0.35          # text usually carries explicit intent
FUSED_FLOOR = 0.28                     # below -> uncertain response


class AirportAssistant:
    def __init__(self):
        self.clf = load_intent_classifier()
        self.retriever = KBRetriever()
        self.vmodel, self.vclasses = load_model()
        self.vindex, self.vmeta = load_index()

    # ---------------- single-modality experts ----------------
    def process_image(self, pil_img):
        return classify_image(pil_img, self.vmodel, self.vclasses,
                              self.vindex, self.vmeta)

    def process_audio(self, wav_path):
        from speech import written_form
        text = transcribe(wav_path)
        text = written_form(text) if text else text   # spoken -> written form
        return {"transcript": text, "nlu": understand(text, self.clf) if text else None}

    def process_text(self, text):
        return understand(text, self.clf)

    # ---------------- fusion ----------------
    def _image_scores(self, vision):
        """Distribute classifier confidence over KB records of matching category."""
        scores = np.zeros(len(self.retriever.records))
        for t in vision["top3"]:
            for i, r in enumerate(self.retriever.records):
                if r["category"] == t["category"]:
                    scores[i] = max(scores[i], t["prob"])
        return scores

    def _text_scores(self, text, nlu):
        hits = self.retriever.search(text, nlu["intent"], nlu["entities"],
                                     k=len(self.retriever.records))
        scores = np.zeros(len(self.retriever.records))
        id2i = {r["id"]: i for i, r in enumerate(self.retriever.records)}
        for h in hits:
            scores[id2i[h["record"]["id"]]] = h["score"]
        return scores

    def respond(self, text=None, image=None, audio_path=None):
        """Main entry point: any combination of typed text, PIL image, wav path."""
        details = {"modalities": []}
        vision = nlu = None
        transcript = None

        if audio_path is not None:
            details["modalities"].append("voice")
            a = self.process_audio(audio_path)
            transcript = a["transcript"]
            details["transcript"] = transcript
            if a["nlu"]:
                nlu = a["nlu"]
            text = (text + " " + transcript).strip() if text else transcript
        if text:
            if "voice" not in details["modalities"]:
                details["modalities"].append("text")
            nlu = self.process_text(text)
            details["nlu"] = nlu
        if image is not None:
            details["modalities"].append("image")
            vision = self.process_image(image)
            details["vision"] = {k: vision[k] for k in ("top3", "category", "confidence")}

        n = len(self.retriever.records)
        t_scores = self._text_scores(text, nlu) if (text and nlu) else np.zeros(n)
        i_scores = self._image_scores(vision) if vision else np.zeros(n)

        if text and vision:                              # true multimodal fusion
            fused = W_TEXT * t_scores + W_IMAGE * i_scores
            mode = "weighted fusion (text %.2f + image %.2f)" % (W_TEXT, W_IMAGE)
        elif text:
            fused = t_scores; mode = "text/voice routing"
        elif vision:
            fused = i_scores * 0.85                      # image alone is weaker evidence
            mode = "image routing"
        else:
            return {"ok": False, "message": "Please provide a photo, a voice "
                    "recording, or a typed question.", "details": details}

        best = int(np.argmax(fused)); conf = float(fused[best])
        details["fusion_mode"] = mode
        details["confidence"] = round(conf, 3)

        low_text_conf = nlu is not None and nlu["confidence"] < CONFIDENCE_FLOOR and not vision
        if conf < FUSED_FLOOR or (transcript == "" and not text and not vision) or low_text_conf:
            return {"ok": False, "confidence": round(conf, 3), "details": details,
                    "message": "I am not confident enough to answer that reliably. "
                               "Please rephrase, or ask a member of staff / the "
                               "information desk (Level 1, Departures Hall). For "
                               "live flight information always check the official "
                               "airport screens."}

        r = self.retriever.records[best]
        msg = (f"**{r['name']}** ({r['terminal']}, {r['floor_zone']})\n\n"
               f"{r['description']}\n\n"
               f"Directions: {r['directions']}\n"
               f"Opening hours: {r['opening_hours']}\n"
               f"Accessibility: {r['accessibility']}")
        if nlu and nlu["intent"] == "flight_status":
            msg += ("\n\nNote: I cannot access live flight data. Please verify "
                    "boarding and delay information on the official departure "
                    "screens or with your airline.")
        return {"ok": True, "record": r, "confidence": round(conf, 3),
                "message": msg, "details": details}
