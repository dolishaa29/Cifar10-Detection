# streamlit_app.py
import streamlit as st
import requests
from PIL import Image
import io
import os

st.set_page_config(page_title="CIFAR-10 Predictor", layout="centered")

st.title("CIFAR-10 Image Classifier (Streamlit → Flask)")

# Edit this depending on where your Flask API runs
FLASK_URL = os.environ.get("FLASK_URL", "http://localhost:5000")
PREDICT_ENDPOINT = FLASK_URL.rstrip("/") + "/predict"

uploaded_file = st.file_uploader("Upload an image (jpg/png). We'll resize to 32x32 for CIFAR-10.", type=["png","jpg","jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded image", use_column_width=True)
    if st.button("Predict"):
        # send file to Flask
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        try:
            with st.spinner("Sending image to Flask API..."):
                resp = requests.post(PREDICT_ENDPOINT, files=files, timeout=20)
            if resp.status_code != 200:
                st.error(f"API error: {resp.status_code} {resp.text}")
            else:
                data = resp.json()
                st.success(f"Predicted: **{data['predicted_class']}** (index {data['predicted_index']})")
                st.subheader("Probabilities")
                probs = data.get("probabilities", {})
                for cls, p in probs.items():
                    st.write(f"- {cls}: {p:.3f}")
        except Exception as e:
            st.error(f"Failed to call API: {e}")
