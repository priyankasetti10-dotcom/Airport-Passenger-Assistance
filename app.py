"""
Smart Airport Passenger Assistance Chatbot - Streamlit prototype (Task 6).

Run:  streamlit run app/app.py
Inputs: image upload, audio upload (wav), typed text - alone or combined.
"""
import os, sys, io, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
from PIL import Image

st.set_page_config(page_title="Airport Assistant", page_icon="🛫", layout="wide")


@st.cache_resource(show_spinner="Loading multimodal models...")
def get_bot():
    from fusion import AirportAssistant
    return AirportAssistant()


bot = get_bot()

st.title("🛫 Smart Airport Passenger Assistant")
st.caption("Multimodal chatbot - ask by **text**, **voice**, or a **photo** of an airport sign "
           "(Riverton International, fictional). Not connected to live flight data.")

left, right = st.columns([1, 1.2], gap="large")

with left:
    st.subheader("Your question")
    img_file = st.file_uploader("Photo of a sign / board (optional)",
                                type=["jpg", "jpeg", "png"], key="img")
    if img_file:
        st.image(img_file, caption="Uploaded image", width=260)
    aud_file = st.file_uploader("Voice question - WAV audio (optional)",
                                type=["wav"], key="aud")
    if aud_file:
        st.audio(aud_file)
    text = st.text_input("Typed question (optional)",
                         placeholder="e.g. Where is gate B12?")
    ask = st.button("Ask the assistant", type="primary", use_container_width=True)

with right:
    st.subheader("Assistant response")
    if ask:
        image = Image.open(io.BytesIO(img_file.getvalue())) if img_file else None
        audio_path = None
        if aud_file:
            tf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tf.write(aud_file.getvalue()); tf.close()
            audio_path = tf.name
        with st.spinner("Thinking..."):
            res = bot.respond(text=text or None, image=image, audio_path=audio_path)
        if audio_path:
            os.unlink(audio_path)

        if res["ok"]:
            st.success(res["message"])
        else:
            st.warning(res["message"])

        conf = res.get("confidence", res["details"].get("confidence"))
        if conf is not None:
            st.progress(min(1.0, float(conf)),
                        text=f"Retrieval confidence: {conf:.2f} "
                             f"({'answered' if res['ok'] else 'below threshold - not answered'})")

        d = res["details"]
        with st.expander("How this answer was produced", expanded=True):
            st.markdown(f"**Modalities used:** {', '.join(d['modalities']) or '-'}  \n"
                        f"**Fusion mode:** {d.get('fusion_mode', '-')}")
            if d.get("transcript") is not None:
                st.markdown(f"**Speech transcript:** “{d['transcript']}”")
            if d.get("nlu"):
                n = d["nlu"]
                st.markdown(f"**Intent:** `{n['intent']}` (p={n['confidence']:.2f})  \n"
                            f"**Entities:** `{n['entities'] or '{}'}`")
            if d.get("vision"):
                v = d["vision"]
                st.markdown(f"**Sign category:** `{v['category']}` "
                            f"(p={v['confidence']:.2f}; top-3: "
                            + ", ".join(f"{t['category']} {t['prob']:.2f}" for t in v["top3"]) + ")")
    else:
        st.info("Upload a photo, add a voice recording, or type a question - "
                "then press **Ask the assistant**.")

st.divider()
st.caption("⚠️ Privacy: photos and voice recordings are processed on this device only and are "
           "not stored after the answer is produced. For live gate, boarding or delay "
           "information always check official airport screens or staff. "
           "Accessibility help: Special Assistance Desk, Level 1 (24 h line +44 20 7946 0110).")
