import streamlit as st
import requests
import pickle
import numpy as np

# -----------------------------
# LOAD CLASSICAL MODELS
# -----------------------------
@st.cache_resource
def load_classical():
    vectorizer = pickle.load(open("saved_model/vectorizer.pkl", "rb"))
    scaler = pickle.load(open("saved_model/scaler.pkl", "rb"))
    model = pickle.load(open("saved_model/culture_features.pkl", "rb"))
    return vectorizer, scaler, model

vectorizer, scaler, rf_model = load_classical()

# -----------------------------
# HUGGINGFACE TOKEN
# -----------------------------
HF_TOKEN = st.secrets["HF_TOKEN"]

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

# -----------------------------
# HF API FUNCTION (FIXED)
# -----------------------------
def hf_predict(api_url, text):
    payload = {"inputs": text}
    response = requests.post(api_url, headers=headers, json=payload)

    if response.status_code != 200:
        raise Exception(response.text)

    result = response.json()

    # Handle output safely
    if isinstance(result, list):
        label = result[0]["label"]
    else:
        label = result["label"]

    # ✅ FIXED LABEL MAPPING
    prediction = 0 if "REAL" in label.upper() else 1
    return prediction


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Fake News Detection", layout="centered")

st.title("📰 Fake News Detection System")
st.write("B.Sc. Final Year Project | 2026")

text_input = st.text_area("Enter News Text")

model_choice = st.selectbox(
    "Select Model",
    ["Random Forest", "MuRIL", "mBERT", "XLM-RoBERTa"]
)

if st.button("Predict"):

    if text_input.strip() == "":
        st.warning("Please enter some text")
    else:
        try:

            # -----------------------------
            # RANDOM FOREST
            # -----------------------------
            if model_choice == "Random Forest":
                vec = vectorizer.transform([text_input]).toarray()
                vec = scaler.transform(vec)
                pred = rf_model.predict(vec)[0]

            # -----------------------------
            # MuRIL (WORKING)
            # -----------------------------
            elif model_choice == "MuRIL":
                pred = hf_predict(
                    "https://api-inference.huggingface.co/pipeline/text-classification/google/muril-base-cased",
                    text_input
                )

            # -----------------------------
            # mBERT (FIXED ENDPOINT)
            # -----------------------------
            elif model_choice == "mBERT":
                pred = hf_predict(
                    "https://api-inference.huggingface.co/pipeline/text-classification/riddhi04/mbert-hybrid-model",
                    text_input
                )

            # -----------------------------
            # XLM-R (FIXED ENDPOINT)
            # -----------------------------
            elif model_choice == "XLM-RoBERTa":
                pred = hf_predict(
                    "https://api-inference.huggingface.co/pipeline/text-classification/riddhi04/xlmr-hybrid-model",
                    text_input
                )

            # -----------------------------
            # OUTPUT
            # -----------------------------
            if pred == 1:
                st.error("🚨 Fake News Detected")
            else:
                st.success("✅ Real News")

        except Exception as e:
            st.error("Error occurred during prediction")
            st.write(e)
