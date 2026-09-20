"""
Speech-to-text component (Task 4): CMU PocketSphinx, domain-adapted.

Design: an airport kiosk must work offline (passenger audio never leaves the
device - GDPR data minimisation). PocketSphinx provides offline decoding; its
open-vocabulary accuracy on short queries is weak, so we adapt it to the
domain with (a) a JSGF grammar compiled from the passenger-query corpus,
(b) a custom pronunciation dictionary for airport terms (e.g. 'airside'),
and (c) audio conditioning (16 kHz mono, silence padding, light dither so
cepstral mean normalisation initialises correctly on very clean audio).
If grammar decoding yields no hypothesis, we fall back to open dictation and
flag low confidence. OpenAI Whisper is a drop-in replacement when available.
"""
import os, re
import numpy as np
import pandas as pd
import librosa
from num2words import num2words
from pocketsphinx import Config, Decoder

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODELDIR = os.path.join(os.path.dirname(__import__("pocketsphinx").__file__), "model", "en-us")
GRAM = os.path.join(ROOT, "outputs", "airport.gram")
DICT = os.path.join(ROOT, "outputs", "airport.dict")

EXTRA_PRON = {"skyview": "S K AY V Y UW", "halal": "HH AH L AA L",
              "airside": "EH R S AY D"}


def spoken_form(text):
    """Normalise written text to its spoken form (B12 -> b twelve)."""
    t = text.lower().replace("-", " ")
    t = re.sub(r"[^\w\s']", " ", t)
    out = []
    for w in t.split():
        m = re.fullmatch(r"([a-z]+)(\d+)", w)
        if m and len(m.group(1)) <= 2:
            out += list(m.group(1))
            out.append(num2words(int(m.group(2))).replace("-", " ").replace(" and ", " "))
        elif w.isdigit():
            out.append(num2words(int(w)).replace("-", " ").replace(" and ", " "))
        else:
            out.append(w)
    return re.sub(r"\s+", " ", " ".join(out)).strip()


_UNITS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
          "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
          "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
          "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}


def _consume_number(words, i):
    """Parse a number word sequence starting at i; return (value, next_i) or None."""
    val = 0; j = i; seen = False
    if j < len(words) and words[j] in _UNITS and _UNITS[words[j]] < 10 and \
            j + 1 < len(words) and words[j + 1] == "hundred":
        val = _UNITS[words[j]] * 100; j += 2; seen = True
        if j < len(words) and words[j] == "and":
            j += 1
    if j < len(words) and words[j] in _TENS:
        val += _TENS[words[j]]; j += 1; seen = True
        if j < len(words) and words[j] in _UNITS and _UNITS[words[j]] < 10:
            val += _UNITS[words[j]]; j += 1
    elif j < len(words) and words[j] in _UNITS:
        val += _UNITS[words[j]]; j += 1; seen = True
    return (val, j) if seen else None


def written_form(text):
    """Inverse text normalisation: spoken transcript -> written query form.

    'where is gate b twelve'      -> 'where is gate b12'
    'desks thirty to forty'       -> 'desks 30 to 40'
    'flight r a two hundred four' -> 'flight ra204'
    Ensures transcribed voice queries enter the SAME NLP pipeline as typed text.
    """
    words = text.lower().split()
    out = []; i = 0
    while i < len(words):
        w = words[i]
        # single letter(s) followed by a number -> alphanumeric code
        if len(w) == 1 and w.isalpha():
            letters = [w]; j = i + 1
            while j < len(words) and len(words[j]) == 1 and words[j].isalpha():
                letters.append(words[j]); j += 1
            num = _consume_number(words, j)
            if num:
                out.append("".join(letters) + str(num[0])); i = num[1]; continue
        num = _consume_number(words, i)
        if num:
            out.append(str(num[0])); i = num[1]; continue
        out.append(w); i += 1
    return " ".join(out)


def build_domain_resources():
    """Compile JSGF grammar + extended dictionary from the query corpus."""
    df = pd.read_csv(os.path.join(ROOT, "data", "text", "queries.csv"))
    alts = " | ".join(f"( {spoken_form(q)} )" for q in df["text"].unique())
    with open(GRAM, "w") as f:
        f.write("#JSGF V1.0;\ngrammar airport;\npublic <query> = " + alts + " ;\n")
    with open(DICT, "w") as f:
        f.write(open(os.path.join(MODELDIR, "cmudict-en-us.dict")).read())
        for w, p in EXTRA_PRON.items():
            f.write(f"{w} {p}\n")
    return GRAM, DICT


class SphinxTranscriber:
    def __init__(self):
        if not (os.path.exists(GRAM) and os.path.exists(DICT)):
            build_domain_resources()
        self.grammar_dec = Decoder(Config(hmm=os.path.join(MODELDIR, "en-us"),
                                          dict=DICT, jsgf=GRAM, lm=None,
                                          logfn="/dev/null"))
        self.open_dec = Decoder(Config(hmm=os.path.join(MODELDIR, "en-us"),
                                       dict=DICT,
                                       lm=os.path.join(MODELDIR, "en-us.lm.bin"),
                                       logfn="/dev/null"))
        self._rng = np.random.RandomState(0)

    def _condition(self, path):
        y, _ = librosa.load(path, sr=16000, mono=True)
        pad = np.zeros(int(0.4 * 16000), dtype=np.float32)
        y = np.concatenate([pad, y, pad])
        y = y + self._rng.normal(0, 10 ** (-45 / 20), len(y)).astype(np.float32)
        return (np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes()

    def _decode(self, dec, raw):
        dec.start_utt(); dec.process_raw(raw, False, True); dec.end_utt()
        return dec.hyp().hypstr.strip() if dec.hyp() else ""

    def transcribe(self, path):
        """Returns (text, mode) where mode is 'grammar' | 'open' | 'none'."""
        raw = self._condition(path)
        hyp = self._decode(self.grammar_dec, raw)
        if hyp:
            return hyp, "grammar"
        hyp = self._decode(self.open_dec, raw)
        return (hyp, "open") if hyp else ("", "none")


_singleton = None


def transcribe(path):
    global _singleton
    if _singleton is None:
        _singleton = SphinxTranscriber()
    return _singleton.transcribe(path)
