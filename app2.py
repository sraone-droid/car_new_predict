import streamlit as st
import joblib
import numpy as np
import os

# -------------------- PAGE UI --------------------
st.set_page_config(page_title="Car Issue Predictor", layout="centered")

BACKGROUND = "https://cdn.dribbble.com/userupload/22797976/file/original-3b362f19987e09fbeb2b092dc029db17.gif"

st.markdown(f"""
<style>
.stApp {{
  background-image: url("{BACKGROUND}");
  background-size: cover;
  background-attachment: fixed;
  color: white;
}}
h1{{text-align:center;color:white;text-shadow:2px 2px 8px black;}}
h3{{color:white;}}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>CAR ISSUE PREDICTOR</h1>", unsafe_allow_html=True)

# -------------------- LOAD MODEL --------------------
VEC = "vectorizer.pkl"
MOD = "model.pkl"

if not os.path.exists(VEC): VEC = "./vectorizer.pkl"
if not os.path.exists(MOD): MOD = "./model.pkl"

vectorizer = joblib.load(VEC)
model = joblib.load(MOD)

# -------------------- SUGGESTED ACTIONS --------------------
SUGGESTIONS = {
    "Weak battery": "Check battery voltage, clean terminals, try jump start. Replace if old.",
    "Faulty spark plug": "Check spark plugs & wiring. Replace worn plugs.",
    "Brake issue": "Inspect brake pads, discs, and fluid level. Avoid high speed driving.",
    "Radiator leak": "Check coolant level, hoses, radiator cap. Stop driving if overheating.",
    "Unbalanced wheels": "Get wheel balancing and alignment at a workshop.",
    "Alternator issue": "Check charging system. Look for battery warning light.",
    "Fuel pump problem": "Check fuel pressure. Avoid running with low fuel.",
    "Engine overheating": "Check coolant, radiator fan, thermostat.",
    "Gearbox issue": "Check transmission oil level. Drive gently and visit workshop.",
    "Suspension issue": "Drive slowly over bumps. Inspect shock absorbers.",
}

DEFAULT_SUGGESTION = "Visit a workshop for diagnosis."

# -------------------- INPUT --------------------
st.subheader("Describe the car issue:")
text = st.text_area("Eg: engine cranks but won’t start, burning smell, vibration…", height=150)

# -------------------- PREDICT --------------------
if st.button("🔍 Predict Issue"):

    text = text.strip()

    if len(text) < 3:
        st.error("❌ Please enter a valid complaint.")
        st.stop()

    vec = vectorizer.transform([text])
    nnz = vec.nnz

    if nnz < 2:
        st.error("❌ Input too vague or contains unknown words. Describe real symptoms.")
        st.stop()

    probs = model.predict_proba(vec)[0]
    best_index = np.argmax(probs)
    best_class = model.classes_[best_index]
    best_prob = probs[best_index]

    if best_prob < 0.45:
        st.error("❌ Low confidence. Please add more details.")
        st.stop()

    # ---------------- RESULT ----------------
    st.success(f"🔧 Likely Issue: **{best_class}** ({best_prob*100:.1f}%)")

    # Suggested action
    suggestion = SUGGESTIONS.get(best_class, DEFAULT_SUGGESTION)

    st.info(f"🛠️ Suggested Action: **{suggestion}**")

    # ---------------- TOP 3 ----------------
    st.markdown("### 🔝 Top 3 Possible Issues:")
    top3 = probs.argsort()[::-1][:3]

    for i in top3:
        c = model.classes_[i]
        p = probs[i] * 100
        st.write(f"• {c} ({p:.1f}%)")
