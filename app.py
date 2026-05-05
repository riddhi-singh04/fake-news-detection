import streamlit as st
import joblib
import numpy as np
import os
import gdown
import torch
import re
import requests

from gradio_client import Client

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Fake News Detection", layout="wide")

# -------------------------------
# THEME
# -------------------------------
theme = st.sidebar.selectbox("Choose Theme", ["Dark", "Light", "Smooth"])

if theme == "Dark":
    st.markdown("""
        <style>
        .stApp {background-color:#0e1117;color:white;}
        </style>
    """, unsafe_allow_html=True)

elif theme == "Light":
    st.markdown("""
        <style>
        .stApp {background-color:white;color:black;}
        </style>
    """, unsafe_allow_html=True)

elif theme == "Smooth":
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(to right,#0f2027,#203a43,#2c5364);
            color:white;
        }
        </style>
    """, unsafe_allow_html=True)

# -------------------------------
# LOAD CLASSICAL MODEL
# -------------------------------
model = joblib.load("saved_model/model.pkl")
vectorizer = joblib.load("saved_model/vectorizer.pkl")
scaler = joblib.load("saved_model/scaler.pkl")
culture_features = joblib.load("saved_model/culture_features.pkl")

# -------------------------------
# UI HEADER
# -------------------------------
st.title("📰 Fake News Verification Engine")
st.subheader("AI-powered verification with Cultural Context Analysis")
st.divider()

# -------------------------------
# MODEL SELECT
# -------------------------------
model_choice = st.selectbox(
    "Select Model",
    ["RandomForest", "mBERT", "XLM-RoBERTa", "MuRIL"]
)

# -------------------------------
# EXAMPLES
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    if st.button("Load Real Example"):
        st.session_state.text = "RBI revised repo rates to control inflation."

with col2:
    if st.button("Load Fake Example"):
        st.session_state.text = "Drinking bleach cures all diseases instantly."

# -------------------------------
# INPUT
# -------------------------------
text = st.text_area("Enter News", value=st.session_state.get("text",""))

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
# SAFE HF CALL (IMPORTANT FIX)
# -------------------------------
def hf_predict(api_url, text, token):
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    response = requests.post(api_url, headers=headers, json={"inputs": text})

    if response.status_code != 200:
        raise Exception(response.text)

    try:
        result = response.json()
    except:
        raise Exception("Invalid response from model")

    if not result:
        raise Exception("Empty response")

    if isinstance(result, list):
        result = result[0]

    label = result.get("label", "REAL")
    score = result.get("score", 0.5)

    pred = 1 if "FAKE" in label.upper() else 0

    return pred, score

# -------------------------------
# PREDICT BUTTON
# -------------------------------
if st.button("Predict"):

    if text.strip() == "":
        st.warning("Enter some text")
    else:
        try:

            # --------------------------
            # RANDOM FOREST
            # --------------------------
            if model_choice == "RandomForest":

                vec = vectorizer.transform([text])
                cult = scaler.transform(get_culture_features(text))

                final = np.hstack((vec.toarray(), cult))

                pred = model.predict(final)[0]
                prob = model.predict_proba(final)[0]

                prediction = int(pred)
                confidence = prob[pred]

            # --------------------------
            # mBERT
            # --------------------------
            elif model_choice == "mBERT":

                prediction, confidence = hf_predict(
                    "https://api-inference.huggingface.co/models/riddhi04/mbert-hybrid-model",
                    text,
                    st.secrets["HF_TOKEN_Riddhi"]
                )

            # --------------------------
            # XLM-R
            # --------------------------
            elif model_choice == "XLM-RoBERTa":

                prediction, confidence = hf_predict(
                    "https://api-inference.huggingface.co/models/riddhi04/xlmr-hybrid-model",
                    text,
                    st.secrets["HF_TOKEN_Riddhi"]
                )

            # --------------------------
            # MURIL (UNCHANGED)
            # --------------------------
            elif model_choice == "MuRIL":

                client = Client(
                    "pseudokoo/FakeNews-Detector-1-API",
                    token=st.secrets["HF_TOKEN"]
                )

                result = client.predict(
                    text=text,
                    api_name="/predict_fake_news"
                )

                result = str(result)

                prediction = 1 if "FAKE" in result.upper() else 0
                confidence = 0.85

            # --------------------------
            # OUTPUT
            # --------------------------
            if prediction == 1:
                st.error("🚨 FAKE NEWS")
            else:
                st.success("✅ REAL NEWS")

            st.write(f"Confidence: {round(confidence*100,2)}%")

        except Exception as e:
            st.error("Error occurred during prediction")
            st.write(str(e))

st.divider()
st.caption("B.Sc. Final Year Project | 2026")
