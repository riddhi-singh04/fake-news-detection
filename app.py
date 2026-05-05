import streamlit as st
import joblib
import numpy as np
import os
import gdown
import torch
import re
import requests

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from gradio_client import Client

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Fake News Detection", layout="wide")

# -------------------------------
# SESSION STATE
# -------------------------------
if "user_text" not in st.session_state:
    st.session_state.user_text = ""

# -------------------------------
# LOAD CLASSICAL MODELS
# -------------------------------
model = joblib.load("saved_model/model.pkl")
vectorizer = joblib.load("saved_model/vectorizer.pkl")
scaler = joblib.load("saved_model/scaler.pkl")
culture_features = joblib.load("saved_model/culture_features.pkl")

# -------------------------------
# UI
# -------------------------------
st.title("📰 Fake News Verification Engine")

model_choice = st.selectbox(
    "Select Model",
    ["RandomForest", "mBERT", "XLM-RoBERTa", "MuRIL"]
)

st.subheader("Try Example News")

col1, col2 = st.columns(2)

with col1:
    if st.button("Load Real News Example"):
        st.session_state.user_text = "The Reserve Bank of India announced a revision in repo rates."

with col2:
    if st.button("Load Fake News Example"):
        st.session_state.user_text = "Scientists confirm drinking bleach cures all diseases instantly."

user_text = st.text_area("Enter News Text", value=st.session_state.user_text)

# -------------------------------
# CULTURE FEATURES
# -------------------------------
def get_culture_features(text):
    text = text.lower()

    features = {f: 0 for f in culture_features}

    if "india" in text:
        features["national_identity_count"] += 1
    if "cure" in text:
        features["health_belief_count"] += 1

    return np.array([features[f] for f in culture_features]).reshape(1, -1)

# -------------------------------
# TRANSFORMER PREDICT
# -------------------------------
def transformer_predict(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        conf = probs[0][pred].item()

    return pred, conf

# -------------------------------
# PREDICTION BUTTON (BOTTOM)
# -------------------------------
if st.button("Predict"):

    if user_text.strip() == "":
        st.warning("Please enter text")
    else:
        try:

            # --------------------------
            # RANDOM FOREST
            # --------------------------
            if model_choice == "RandomForest":

                vec = vectorizer.transform([user_text])
                cult = get_culture_features(user_text)
                cult = scaler.transform(cult)

                final = np.hstack((vec.toarray(), cult))

                pred = model.predict(final)[0]
                prob = model.predict_proba(final)[0]

                prediction = int(pred)
                confidence = prob[pred]

            # --------------------------
            # mBERT
            # --------------------------
            elif model_choice == "mBERT":

                token = st.secrets["HF_TOKEN_Riddhi"]

                headers = {
                    "Authorization": f"Bearer {token}"
                }

                response = requests.post(
                    "https://api-inference.huggingface.co/models/riddhi04/mbert-hybrid-model",
                    headers=headers,
                    json={"inputs": user_text}
                )

                result = response.json()

                if isinstance(result, list):
                    result = result[0]

                label = result.get("label", "REAL")
                score = result.get("score", 0.5)

                prediction = 1 if "FAKE" in label.upper() else 0
                confidence = score

            # --------------------------
            # XLM-R
            # --------------------------
            elif model_choice == "XLM-RoBERTa":

                token = st.secrets["HF_TOKEN_Riddhi"]

                headers = {
                    "Authorization": f"Bearer {token}"
                }

                response = requests.post(
                    "https://api-inference.huggingface.co/models/riddhi04/xlmr-hybrid-model",
                    headers=headers,
                    json={"inputs": user_text}
                )

                result = response.json()

                if isinstance(result, list):
                    result = result[0]

                label = result.get("label", "REAL")
                score = result.get("score", 0.5)

                prediction = 1 if "FAKE" in label.upper() else 0
                confidence = score

            # --------------------------
            # MURIL
            # --------------------------
            elif model_choice == "MuRIL":

                token = st.secrets["HF_TOKEN"]

                client = Client(
                    "pseudokoo/FakeNews-Detector-1-API",
                    token=token
                )

                result = client.predict(
                    text=user_text,
                    api_name="/predict_fake_news"
                )

                result = str(result)

                prediction = 1 if "FAKE" in result else 0
                confidence = 0.85

            # --------------------------
            # OUTPUT
            # --------------------------
            if prediction == 1:
                st.error("🚨 FAKE NEWS")
            else:
                st.success("✅ REAL NEWS")

            st.write(f"Confidence: {round(confidence * 100, 2)}%")

        except Exception as e:
            st.error("Error occurred during prediction")
            st.write(str(e))
