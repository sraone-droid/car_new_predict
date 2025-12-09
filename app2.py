import streamlit as st
import joblib
import numpy as np
import os
import datetime
import json

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

if not os.path.exists(VEC):
    VEC = "./vectorizer.pkl"

if not os.path.exists(MOD):
    MOD = "./model.pkl"

vectorizer = joblib.load(VEC)
model = joblib.load(MOD)

# -------------------- SUGGESTED ACTIONS --------------------
ACTIONS = {
    "Engine Overheating": "Check coolant level, radiator fan, thermostat. Avoid long driving.",
    "Brake Failure": "Do not drive. Inspect brake pads, fluid level, master cylinder.",
    "Battery Issue": "Check terminals corrosion. Recharge or replace battery.",
    "Oil Leak": "Check under car for leak. Inspect oil pan & gasket.",
    "Suspension Noise": "Slow down on bumps. Inspect shocks & bushings.",
    "Transmission Problem": "Stop hard driving. Check fluid & gearbox.",
    "Unknown": "Not enough detail. Provide more symptoms."
}

# -------------------- INPUT --------------------
st.subheader("Describe the car issue:")
text = st.text_area("Enter symptoms, sounds, behavior…", height=150)

# -------------------- PREDICT --------------------
if st.button("🔍 Predict Issue"):

    # --- CHECKS ---
    if len(text.strip()) < 3:
        st.error("❌ Please enter a meaningful description.")
        st.stop()

    vec = vectorizer.transform([text])
    nnz = vec.nnz

    if nnz < 2:
        st.error("❌ Input too vague or contains unknown words. Describe real car symptoms.")
        st.stop()

    probs = model.predict_proba(vec)[0]
    best_prob = np.max(probs)
    best_class = model.classes_[np.argmax(probs)]

    if best_prob < 0.40:
        st.error("❌ Not enough confidence to predict. Add more detail about the issue.")
        st.stop()

    # -------------------- SHOW RESULT --------------------
    st.success(f"🔧 Likely Issue: **{best_class}** ({best_prob*100:.1f}% confidence)")

    # suggested action
    action = ACTIONS.get(best_class, "No info available")
    st.info(f"👉 Suggested Action: {action}")

    # -------------------- FEEDBACK SECTION --------------------
    st.subheader("🛠️ Feedback")

    feedback = st.radio(
        "Was this prediction correct?",
        ["Yes", "No"],
        horizontal=True
    )

    comment = st.text_input("Any comments?", "")

    if st.button("Submit Feedback"):

        log_data = {
            "timestamp": str(datetime.datetime.now()),
            "input": text,
            "prediction": best_class,
            "correct": feedback,
            "comment": comment
        }

        with open("feedback_log.json", "a") as f:
            f.write(json.dumps(log_data) + "\n")

        st.success("Thank you! Feedback recorded 👍")

# -------------------- SHOW FEEDBACK COUNT --------------------
st.divider()

if os.path.exists("feedback_log.json"):
    with open("feedback_log.json") as f:
        count = len(f.readlines())
    st.caption(f"📊 Total feedback collected: **{count}**")
else:
    st.caption("📊 No feedback yet.")
