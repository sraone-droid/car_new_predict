import streamlit as st
import joblib
import numpy as np
import os

st.set_page_config(page_title="Car Issue Predictor", layout="centered")
BG = "https://cdn.dribbble.com/userupload/22797976/file/original-3b362f19987e09fbeb2b092dc029db17.gif"

# Background styling
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

# --------- LOAD MODEL + VECTORIZER SAFELY ----------
def load_artifact(primary, fallback):
    """Return first existing path."""
    return primary if os.path.exists(primary) else fallback

vec_path = load_artifact("vectorizer.pkl", "artifacts/vectorizer.pkl")
model_path = load_artifact("model.pkl", "artifacts/model.pkl")

try:
    vectorizer = joblib.load(vec_path)
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# -------------- APP UI --------------
st.subheader("Describe the car issue:")
user_input = st.text_area("Enter symptoms, sounds, behavior…", height=150)

if st.button("🔍 Predict Issue"):
    if user_input.strip() == "":
        st.warning("Please enter a description of the issue.")
    else:
        # Transform input
        X = vectorizer.transform([user_input])
        pred = model.predict(X)[0]

        st.success(f"### 🔧 Likely Issue: **{pred}**")
