import streamlit as st
import joblib
import numpy as np
import os

# -------------------- PAGE UI --------------------
st.set_page_config(page_title="Car Issue Predictor", layout="centered")

BG = "https://cdn.dribbble.com/userupload/22797976/file/original-3b362f19987e09fbeb2b092dc029db17.gif"

st.markdown(f"""
<style>
.stApp {{
  background-image: url("{BG}");
  background-size: cover;
  background-attachment: fixed;
}}
h1{{text-align:center;color:white;text-shadow:1px 1px 6px rgba(0,0,0,0.7);}}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🚗 Car Issue Predictor</h1>", unsafe_allow_html=True)

# -------------------- LOAD MODEL --------------------
VEC = "vectorizer.pkl"
MOD = "model.pkl"

# if inside Streamlit Cloud, sometimes file lives in repo root or in /mount/src
if not os.path.exists(VEC):
    VEC = "./vectorizer.pkl"

if not os.path.exists(MOD):
    MOD = "./model.pkl"

vectorizer = joblib.load(VEC)
model = joblib.load(MOD)

# -------------------- INPUT --------------------
st.subheader("Describe the car issue:")
text = st.text_area("Enter symptoms, sounds, behavior…", height=150)

# -------------------- PREDICT --------------------
if st.button("🔍 Predict Issue"):

    # --- BASIC CHECKS ---
    if len(text.strip()) < 3:
        st.error("❌ Please enter a meaningful description.")
        st.stop()

    vec = vectorizer.transform([text])
    nnz = vec.nnz  # number of non-zero TF-IDF matches

    # Reject random nonsense like "asgfsafa"
    if nnz < 2:
        st.error("❌ Input too vague or contains unknown words. Describe real car symptoms.")
        st.stop()

    # Get probabilities
    probs = model.predict_proba(vec)[0]
    best_prob = np.max(probs)
    best_class = model.classes_[np.argmax(probs)]

    # Confidence threshold (avoid random guessing)
    if best_prob < 0.40:
        st.error("❌ Not enough confidence to predict. Add more detail about the issue.")
        st.stop()

    # Show result
    st.success(f"🔧 Likely Issue: **{best_class}** ({best_prob*100:.1f}% confidence)")

