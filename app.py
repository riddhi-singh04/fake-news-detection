import streamlit as st
import joblib
import numpy as np
import os
import gdown
import torch
import re
import requests

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from gradio_client import Client

# -------------------------------------------------
# Page configuration
# -------------------------------------------------

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide"
)

# -------------------------------------------------
# Session state
# -------------------------------------------------

if "user_text" not in st.session_state:
    st.session_state.user_text = ""

# -------------------------------------------------
# Theme toggle
# -------------------------------------------------

theme = st.sidebar.selectbox(
    "Choose Theme",
    ["Dark", "Light", "Smooth"]
)

if theme == "Dark":
    st.markdown("<style>.stApp {background-color: #0e1117; color: white;}</style>", unsafe_allow_html=True)

elif theme == "Light":
    st.markdown("<style>.stApp {background-color: #ffffff; color: black;}</style>", unsafe_allow_html=True)

elif theme == "Smooth":
    st.markdown("<style>.stApp {background: linear-gradient(to right,#0f2027,#203a43,#2c5364); color:white;}</style>", unsafe_allow_html=True)

# -------------------------------------------------
# Download classical model
# -------------------------------------------------

MODEL_PATH = "saved_model/model.pkl"

if not os.path.exists(MODEL_PATH):
    os.makedirs("saved_model", exist_ok=True)
    gdown.download(
        "https://drive.google.com/uc?id=1uV2G8hRJF38FFABo-tWDPwtAxsDOmW5s",
        MODEL_PATH,
        quiet=False
    )

# -------------------------------------------------
# Load classical models
# -------------------------------------------------

model = joblib.load("saved_model/model.pkl")
vectorizer = joblib.load("saved_model/vectorizer.pkl")
scaler = joblib.load("saved_model/scaler.pkl")
culture_features = joblib.load("saved_model/culture_features.pkl")

# -------------------------------------------------
# Header
# -------------------------------------------------

st.title("📰 Fake News Verification Engine")
st.subheader("AI-powered verification with Cultural Context Analysis")
st.divider()

# -------------------------------------------------
# Model selector
# -------------------------------------------------

model_choice = st.selectbox(
    "Select Model",
    ["RandomForest", "mBERT", "XLM-RoBERTa", "MuRIL"]
)

# -------------------------------------------------
# Example buttons
# -------------------------------------------------

st.subheader("Try Example News")

col1, col2 = st.columns(2)

with col1:
    if st.button("Load Real News Example"):
        st.session_state.user_text = "The Reserve Bank of India announced a revision in repo rates to control inflation."

with col2:
    if st.button("Load Fake News Example"):
        st.session_state.user_text = "Scientists confirm drinking bleach cures all diseases instantly."

# -------------------------------------------------
# Input
# -------------------------------------------------

user_text = st.text_area(
    "Enter News Text",
    height=180,
    value=st.session_state.user_text
)

st.session_state.user_text = user_text

# -------------------------------------------------
# Cultural features
# -------------------------------------------------

def get_culture_features(text):
    text_lower = text.lower()
    features = {f: 0 for f in culture_features}

    keyword_map = {
        "national_identity_count": ["india", "country", "nation"],
        "health_belief_count": ["cure", "disease", "virus"],
        "community_tension_count": ["riot", "violence", "attack"],
        "education_count": ["school", "college", "university"]
    }

    for name, words in keyword_map.items():
        if name in features:
            features[name] = sum(1 for w in words if w in text_lower)

    return np.array([features[f] for f in culture_features]).reshape(1, -1)

# -------------------------------------------------
# Prediction
# -------------------------------------------------

if st.button("Predict"):

    if user_text.strip() == "":
        st.warning("Please enter news text.")

    else:
        try:

            with st.spinner("Analyzing news content..."):

                # ------------------------------
                # RANDOM FOREST
                # ------------------------------
                if model_choice == "RandomForest":

                    text_vector = vectorizer.transform([user_text])
                    culture_vector = scaler.transform(get_culture_features(user_text))

                    combined = np.hstack((text_vector.toarray(), culture_vector))

                    prediction = int(model.predict(combined)[0])
                    probability = model.predict_proba(combined)[0]
                    confidence = probability[prediction]

                # ------------------------------
                # mBERT (FIXED JSON ERROR)
                # ------------------------------
                elif model_choice == "mBERT":

                    elif model_choice == "mBERT":
                        token = st.secrets["HF_TOKEN_Riddhi"]
                    
                        # Step 1: Verify model slug matches exactly what's on HuggingFace
                        MODEL_ID = "riddhi04/mbert-hybrid-model"  # ← double-check this on hf.co
                    
                        response = requests.post(
                            f"https://api-inference.huggingface.co/models/{MODEL_ID}",
                            headers={
                                "Authorization": f"Bearer {token}",
                                "Content-Type": "application/json"
                            },
                            json={"inputs": user_text},
                            timeout=30
                        )
                    
                        # Step 2: Better error reporting
                        if response.status_code != 200:
                            st.error(f"HF API error {response.status_code}")
                            st.code(response.text[:500])   # show first 500 chars safely
                            st.stop()
                    
                        result = response.json()
                    
                        # Step 3: Handle HF's actual nested-list response format
                        # result = [[{"label": "FAKE", "score": 0.91}, {"label": "REAL", "score": 0.09}]]
                        if isinstance(result, dict) and "error" in result:
                            st.warning(f"Model loading: {result['error']} — wait ~20s and retry")
                            st.stop()
                    
                        # Unwrap nested list
                        if isinstance(result, list) and isinstance(result[0], list):
                            result = result[0]   # now: [{"label": "FAKE", ...}, {"label": "REAL", ...}]
                    
                        # Get the top prediction (highest score)
                        top = max(result, key=lambda x: x["score"])
                        label = top.get("label", "REAL")
                        confidence = top.get("score", 0.5)
                        prediction = 1 if "FAKE" in label.upper() else 0

                # ------------------------------
                # XLM-R (FIXED JSON ERROR)
                # ------------------------------
                elif model_choice == "XLM-RoBERTa":

                    token = st.secrets["HF_TOKEN_Riddhi"]

                    response = requests.post(
                        "https://api-inference.huggingface.co/models/riddhi04/xlmr-hybrid-model",
                        headers={"Authorization": f"Bearer {token}"},
                        json={"inputs": user_text}
                    )

                    if response.status_code != 200:
                        st.error("Model not ready / invalid response")
                        st.write(response.text)
                        st.stop()

                    try:
                        result = response.json()
                    except:
                        st.error("Invalid response from model")
                        st.write(response.text)
                        st.stop()

                    if isinstance(result, dict) and "error" in result:
                        st.error("Model is loading... wait and retry")
                        st.stop()

                    if isinstance(result, list):
                        result = result[0]

                    label = result.get("label", "REAL")
                    score = result.get("score", 0.5)

                    prediction = 1 if "FAKE" in label.upper() else 0
                    confidence = score

                # ------------------------------
                # MuRIL (UNCHANGED)
                # ------------------------------
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

                    result_str = str(result)

                    prediction = 1 if "FAKE" in result_str.upper() else 0

                    match = re.search(r"(\d+(\.\d+)?)%", result_str)
                    confidence = float(match.group(1)) / 100 if match else 0.85

            if prediction == 1:
                st.error("🚨 Prediction: FAKE")
            else:
                st.success("✅ Prediction: REAL")

            st.write(f"Confidence: {round(confidence*100,2)}%")

        except Exception as e:
            st.error("Error occurred during prediction.")
            st.write(str(e))

st.divider()
st.caption("B.Sc. Final Year Project | 2026")
