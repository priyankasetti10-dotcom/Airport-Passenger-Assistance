"""
Passenger query dataset builder (text + voice).

- Builds a labelled text dataset of typed passenger queries: intent + entities.
- Synthesises spoken versions of a subset with espeak-ng (varied voice, speed,
  pitch to simulate different speakers), at 16 kHz WAV, and creates noisy
  variants to stress-test the speech-to-text component.
"""
import csv, os, random, subprocess
import numpy as np
import soundfile as sf
import librosa

random.seed(7); np.random.seed(7)
ROOT = os.path.join(os.path.dirname(__file__), "..")
TXT = os.path.join(ROOT, "data", "text"); AUD = os.path.join(ROOT, "data", "audio")
os.makedirs(TXT, exist_ok=True); os.makedirs(AUD, exist_ok=True)

# intent -> (queries, entities dict per query aligned by index)
DATA = {
    "find_gate": [
        ("Where is gate B12?", {"gate": "B12"}),
        ("How do I get to gate A3 from security?", {"gate": "A3"}),
        ("Which way to gates B1 to B20?", {"gate": "B1-B20"}),
        ("I need to find gate C7 quickly", {"gate": "C7"}),
        ("Can you direct me to my departure gate B12?", {"gate": "B12"}),
        ("gate a3 where", {"gate": "A3"}),
        ("How far is gate B12 from the restaurants?", {"gate": "B12"}),
        ("What is the walking time to gate A3?", {"gate": "A3"}),
    ],
    "baggage_claim": [
        ("How do I get to baggage claim?", {}),
        ("Where is the baggage reclaim hall?", {}),
        ("Which belt has the bags from flight RA204?", {"flight": "RA204"}),
        ("Where do I collect my luggage after landing?", {}),
        ("baggage claim directions please", {}),
        ("My suitcase, where do I pick it up?", {}),
        ("Is baggage claim on the arrivals level?", {"zone": "arrivals"}),
    ],
    "check_in": [
        ("Where is the check in desk for international flights?", {"service": "international check-in"}),
        ("Which desks are for long haul check in?", {}),
        ("Where can I do self check in?", {"service": "self check-in"}),
        ("Where do I drop my bag after online check in?", {}),
        ("check in area terminal 1", {"terminal": "1"}),
        ("What time does check in open?", {}),
        ("Where are check in desks 30 to 40?", {"desk": "30-40"}),
    ],
    "security": [
        ("How long is the queue at security?", {}),
        ("Where is security control?", {}),
        ("Can I take a water bottle through security?", {}),
        ("Where is the fast track security lane?", {"service": "fast track"}),
        ("What are the liquid rules at security?", {}),
        ("security check directions", {}),
    ],
    "restroom": [
        ("Where is the nearest toilet?", {}),
        ("Are there restrooms near gate B8?", {"gate": "B8"}),
        ("Where can I find a baby changing room?", {"service": "baby changing"}),
        ("Is there an accessible toilet on pier B?", {"zone": "Pier B"}),
        ("toilets near the gates", {}),
    ],
    "lounge": [
        ("Is there a lounge near terminal 2?", {"terminal": "2"}),
        ("How do I get into the Skyview lounge?", {"service": "Skyview Lounge"}),
        ("Where is the quiet zone or prayer room?", {"service": "prayer room"}),
        ("Can I pay to enter the lounge?", {}),
        ("Does the lounge have showers?", {}),
    ],
    "transport": [
        ("Where can I find airport transport?", {}),
        ("How do I get the train to the city centre?", {"service": "train"}),
        ("Where is the taxi pickup zone?", {"service": "taxi"}),
        ("When does the last train leave the airport?", {"service": "train"}),
        ("Where do the buses depart from?", {"service": "bus"}),
        ("Where can I rent a car?", {"service": "car rental"}),
        ("Is there a night bus to the city?", {"service": "bus"}),
    ],
    "information": [
        ("Where is the nearest information desk?", {}),
        ("I lost my phone, where is lost and found?", {"service": "lost and found"}),
        ("Who can help me, I need assistance?", {}),
        ("Where can I get a map of the terminal?", {}),
        ("Is there a first aid point in the terminal?", {"service": "first aid"}),
        ("Where do I report a lost bag?", {"service": "lost and found"}),
    ],
    "restaurant": [
        ("Where can I get food before my flight?", {}),
        ("Are there halal food options in the airport?", {"service": "halal"}),
        ("Where is the food court?", {}),
        ("Is there a coffee shop after security?", {}),
        ("What restaurants are open now?", {}),
    ],
    "flight_status": [
        ("Is my flight delayed?", {}),
        ("What is the status of flight RA204?", {"flight": "RA204"}),
        ("Has boarding started for gate B12?", {"gate": "B12"}),
        ("When does boarding close for my flight?", {}),
        ("Why is my flight to Berlin delayed?", {"destination": "Berlin"}),
    ],
    "special_assistance": [
        ("I need wheelchair assistance, where do I go?", {"service": "wheelchair"}),
        ("Where is the special assistance desk?", {}),
        ("My mother has reduced mobility, can someone help her to the gate?", {}),
        ("Do you have help for hidden disabilities?", {}),
        ("Where can I get a sunflower lanyard?", {"service": "sunflower lanyard"}),
    ],
}

EXTRA = {'find_gate': [('Which direction is gate C7?', {'gate': 'C7'}), ('Take me to gate B12 please', {'gate': 'B12'}), ('Is gate A3 far from here?', {'gate': 'A3'}), ('What pier are gates B1 to B20 on?', {'gate': 'B1-B20'}), ('Looking for my boarding gate B12', {'gate': 'B12'})], 'baggage_claim': [('Which way to the luggage belts?', {}), ('I just landed, where are the bags?', {}), ('Belt number for baggage reclaim please', {}), ('Where is luggage collection?', {})], 'check_in': [('Which counter is my airline check in?', {}), ('I need to check in for my flight, where do I go?', {}), ('Where are the check in kiosks?', {}), ('Bag drop location please', {})], 'security': [('Do I need to remove my laptop at security screening?', {}), ('Which level is the security checkpoint on?', {}), ('How do I get through security fastest?', {}), ('Is the security queue long right now?', {})], 'restroom': [('I need the bathroom', {}), ('Closest washroom to the food court?', {}), ('Where are the ladies and gents toilets?', {}), ('Disabled toilet location please', {})], 'lounge': [('Which lounges can I use with priority pass?', {}), ('Is there somewhere quiet to rest before boarding?', {}), ('How much does lounge entry cost?', {}), ('Where is the business lounge in terminal 2?', {'terminal': '2'})], 'transport': [('How do I get to the city from the airport?', {}), ('Where do I catch the express train?', {'service': 'train'}), ('Best way to get downtown from here?', {}), ('Where is the taxi rank outside arrivals?', {'service': 'taxi'}), ('How often do trains run to the city centre?', {'service': 'train'})], 'information': [('Can someone give me directions, is there a help desk?', {}), ('Where do I ask about airport services?', {}), ("I found someone's wallet, where do I hand it in?", {'service': 'lost and found'}), ('Where is the information point in departures?', {})], 'restaurant': [("I'm hungry, what food is there airside?", {}), ('Any vegan restaurants in the terminal?', {}), ('Where can I buy a sandwich and coffee?', {}), ('Which cafes are near the gates?', {})], 'flight_status': [('Can you check if flight RA204 is on time?', {'flight': 'RA204'}), ('What time does my plane actually leave?', {}), ('Is boarding open yet for the Berlin flight?', {'destination': 'Berlin'}), ('Are there any delays this evening?', {})], 'special_assistance': [('I am travelling with a disabled passenger, what support is there?', {}), ('Can I book a wheelchair to the gate?', {'service': 'wheelchair'}), ('Where do passengers with reduced mobility check in?', {}), ('Is there help for deaf passengers?', {})]}
for k, v in EXTRA.items():
    DATA[k] = DATA[k] + v

# spoken subset: first 3 queries of each intent (33 utterances) x voice variants
VOICES = [("en-gb", 150, 45), ("en-us", 170, 50), ("en-gb+f3", 160, 60),
          ("en-us+f4", 145, 70), ("en-gb+m6", 180, 35)]


def synth(text, wavpath, voice, wpm, pitch):
    subprocess.run(["espeak-ng", "-v", voice, "-s", str(wpm), "-p", str(pitch),
                    "-w", wavpath, text], check=True, capture_output=True)
    y, sr = librosa.load(wavpath, sr=16000)  # resample to 16 kHz mono
    sf.write(wavpath, y, 16000)
    return y


def main():
    rows = []
    for intent, items in DATA.items():
        for q, ents in items:
            rows.append({"text": q, "intent": intent,
                         "entities": ";".join(f"{k}={v}" for k, v in ents.items())})
    with open(os.path.join(TXT, "queries.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["text", "intent", "entities"])
        w.writeheader(); w.writerows(rows)
    print(f"Text dataset: {len(rows)} labelled queries, {len(DATA)} intents")

    manifest = []
    i = 0
    for intent, items in DATA.items():
        for q, _ in items[:3]:
            voice, wpm, pitch = VOICES[i % len(VOICES)]
            fn = f"q{i:03d}.wav"
            y = synth(q, os.path.join(AUD, fn), voice, wpm, pitch)
            manifest.append({"file": fn, "text": q, "intent": intent,
                             "voice": voice, "condition": "clean"})
            # noisy variant: additive babble-like noise + gain drop
            noise = np.random.normal(0, 1, len(y))
            noise = librosa.effects.preemphasis(noise) * 0.05 * np.abs(np.sin(np.linspace(0, 20, len(y))))
            yn = 0.8 * y + noise.astype(np.float32)
            fn2 = f"q{i:03d}_noisy.wav"
            sf.write(os.path.join(AUD, fn2), yn, 16000)
            manifest.append({"file": fn2, "text": q, "intent": intent,
                             "voice": voice, "condition": "noisy"})
            i += 1
    with open(os.path.join(AUD, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "text", "intent", "voice", "condition"])
        w.writeheader(); w.writerows(manifest)
    print(f"Voice dataset: {len(manifest)} wav files ({i} utterances x clean+noisy)")


if __name__ == "__main__":
    main()
