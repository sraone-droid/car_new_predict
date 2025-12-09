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

st.markdown("<h1>🚗 Car Issue Predictor</h1>", unsafe_allow_html=True)

# -------------------- LOAD MODEL --------------------
VEC = "vectorizer.pkl"
MOD = "model.pkl"

# Fallback paths
if not os.path.exists(VEC): VEC = "./vectorizer.pkl"
if not os.path.exists(MOD): MOD = "./model.pkl"

try:
    vectorizer = joblib.load(VEC)
    model = joblib.load(MOD)
except Exception as e:
    st.error("❌ Model or vectorizer not found. Ensure both files exist.")
    st.stop()

# -------------------- USER INPUT --------------------
st.subheader("Describe the car issue:")
text = st.text_area("Eg: engine cranks but won’t start, burning smell, vibration…", height=150)

# -------------------- PREDICT --------------------
if st.button("🔍 Predict Issue"):

    text = text.strip()

    # Basic check
    if len(text) < 3:
        st.error("❌ Please enter a valid complaint.")
        st.stop()

    # Vectorize
    vec = vectorizer.transform([text])
    nnz = vec.nnz

    # Reject nonsense
    if nnz < 2:
        st.error("❌ Input too vague or contains unknown words. Describe real symptoms.")
        st.stop()

    # Probability prediction
    probs = model.predict_proba(vec)[0]
    best_index = np.argmax(probs)
    best_class = model.classes_[best_index]
    best_prob = probs[best_index]

    # Confidence threshold
    if best_prob < 0.45:
        st.error("❌ Low confidence. Please add more details.")
        st.stop()

    # MAIN RESULT
    st.success(f"🔧 Likely Issue: **{best_class}** ({best_prob*100:.1f}%)")


    # ---------------- TOP 3 PREDICTIONS ----------------
    st.markdown("### 🔝 Top 3 Possible Issues:")
    top3 = probs.argsort()[::-1][:3]

    for i in top3:
        st.write(f"• {model.classes_[i]} ({probs[i]*100:.1f}%)")
