import streamlit as st
import joblib
import numpy as np
import os
import gdown
import torch
import re

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

    st.markdown(
        """
        <style>
        .stApp {
            background-color: #0e1117;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

elif theme == "Light":

    st.markdown(
        """
        <style>
        .stApp {
            background-color: #ffffff;
            color: black;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

elif theme == "Smooth":

    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(
                to right,
                #0f2027,
                #203a43,
                #2c5364
            );
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# -------------------------------------------------
# Download classical model if missing
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
# Load classical model
# -------------------------------------------------

model = joblib.load("saved_model/model.pkl")
vectorizer = joblib.load("saved_model/vectorizer.pkl")
scaler = joblib.load("saved_model/scaler.pkl")
culture_features = joblib.load(
    "saved_model/culture_features.pkl"
)

# -------------------------------------------------
# Load transformer models (PRIVATE HF)
# -------------------------------------------------

@st.cache_resource
def load_transformers():

    device = torch.device("cpu")

    muril_token = st.secrets["HF_TOKEN"]
    riddhi_token = st.secrets["HF_TOKEN_Riddhi"]

    # -------------------------
    # mBERT
    # -------------------------

    mb_tokenizer = AutoTokenizer.from_pretrained(
        "riddhi04/mbert-hybrid-model",
        token=riddhi_token
    )

    mb_model = AutoModelForSequenceClassification.from_pretrained(
        "riddhi04/mbert-hybrid-model",
        token=riddhi_token,
        ignore_mismatched_sizes=True
        trust_remote_code=True
    )

    mb_model.to(device)
    mb_model.eval()

    # -------------------------
    # XLM-R
    # -------------------------

    xlm_tokenizer = AutoTokenizer.from_pretrained(
        "riddhi04/xlmr-hybrid-model",
        token=riddhi_token
    )

    xlmr_model = AutoModelForSequenceClassification.from_pretrained(
        "riddhi04/xlmr-hybrid-model",
        token=riddhi_token,
        ignore_mismatched_sizes=True
        trust_remote_code=True
    )

    xlm_model.to(device)
    xlm_model.eval()

    return (
        mb_model,
        mb_tokenizer,
        xlm_model,
        xlm_tokenizer,
        muril_token
    )


(
    mb_model,
    mb_tokenizer,
    xlm_model,
    xlm_tokenizer,
    muril_token
) = load_transformers()

# -------------------------------------------------
# Header
# -------------------------------------------------

st.title("📰 Fake News Verification Engine")

st.subheader(
    "AI-powered verification with Cultural Context Analysis"
)

st.divider()

# -------------------------------------------------
# Model selector
# -------------------------------------------------

model_choice = st.selectbox(
    "Select Model",
    [
        "RandomForest",
        "mBERT",
        "XLM-RoBERTa",
        "MuRIL"
    ]
)

# -------------------------------------------------
# Example buttons
# -------------------------------------------------

st.subheader("Try Example News")

col1, col2 = st.columns(2)

with col1:

    if st.button("Load Real News Example"):

        st.session_state.user_text = (
            "The Reserve Bank of India announced a revision "
            "in repo rates to control inflation."
        )

with col2:

    if st.button("Load Fake News Example"):

        st.session_state.user_text = (
            "Scientists confirm drinking bleach cures all diseases instantly."
        )

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

    detected = []

    keyword_map = {

        "national_identity_count": [
            "india","country","nation"
        ],

        "health_belief_count": [
            "cure","disease","virus"
        ],

        "community_tension_count": [
            "riot","violence","attack"
        ],

        "education_count": [
            "school","college","university"
        ]

    }

    for feature_name, keywords in keyword_map.items():

        if feature_name in features:

            count = sum(
                1 for word in keywords
                if word in text_lower
            )

            features[feature_name] = count

            if count > 0:
                detected.append(feature_name)

    for key in features:

        if key.startswith("has_"):

            base = key.replace(
                "has_", ""
            ) + "_count"

            if base in features:

                features[key] = int(
                    features[base] > 0
                )

                if features[key] == 1:
                    detected.append(key)

    vector = np.array(
        [features[f] for f in culture_features]
    ).reshape(1, -1)

    return vector, detected

# -------------------------------------------------
# Transformer prediction
# -------------------------------------------------

def transformer_predict(
    text,
    model,
    tokenizer
):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():

        outputs = model(**inputs)

        probs = torch.softmax(
            outputs.logits,
            dim=1
        )

        prediction = torch.argmax(
            probs,
            dim=1
        ).item()

        confidence = probs[
            0,
            prediction
        ].item()

    return prediction, confidence

# -------------------------------------------------
# Prediction
# -------------------------------------------------

if st.button("Predict"):

    if st.session_state.user_text.strip() == "":

        st.warning("Please enter news text.")

    else:

        try:

            with st.spinner(
                "Analyzing news content..."
            ):

                # -------------------------
                # RandomForest
                # -------------------------

                if model_choice == "RandomForest":

                    text_vector = vectorizer.transform(
                        [st.session_state.user_text]
                    )

                    culture_vector, detected_features = \
                        get_culture_features(
                            st.session_state.user_text
                        )

                    culture_vector = scaler.transform(
                        culture_vector
                    )

                    combined = np.hstack(
                        (
                            text_vector.toarray(),
                            culture_vector
                        )
                    )

                    prediction = int(
                        model.predict(combined)[0]
                    )

                    probability = model.predict_proba(
                        combined
                    )[0]

                    confidence = probability[
                        prediction
                    ]

                # -------------------------
                # mBERT
                # -------------------------

                elif model_choice == "mBERT":

                    prediction, confidence = \
                        transformer_predict(
                            st.session_state.user_text,
                            mb_model,
                            mb_tokenizer
                        )

                    detected_features = []

                # -------------------------
                # XLM-R
                # -------------------------

                elif model_choice == "XLM-RoBERTa":

                    prediction, confidence = \
                        transformer_predict(
                            st.session_state.user_text,
                            xlm_model,
                            xlm_tokenizer
                        )

                    detected_features = []

                # -------------------------
                # MuRIL API
                # -------------------------

                elif model_choice == "MuRIL":

                    client = Client(
                        "pseudokoo/FakeNews-Detector-1-API",
                        hf_token=muril_token
                    )

                    result = client.predict(
                        text=st.session_state.user_text,
                        api_name="/predict_fake_news"
                    )

                    result_str = str(result)

                    if "FAKE" in result_str.upper():
                        prediction = 1
                    else:
                        prediction = 0

                    match = re.search(
                        r"(\d+(\.\d+)?)%",
                        result_str
                    )

                    if match:
                        confidence = float(
                            match.group(1)
                        ) / 100
                    else:
                        confidence = 0.85

                    detected_features = []

            if prediction == 1:

                st.error("🚨 Prediction: FAKE")

            else:

                st.success("✅ Prediction: REAL")

            st.write(
                f"Confidence: {round(confidence*100,2)}%"
            )

        except Exception as e:

            st.error(
                "Error occurred during prediction."
            )

            st.write(str(e))

# -------------------------------------------------
# Footer
# -------------------------------------------------

st.divider()

st.caption(
    "B.Sc. Final Year Project | 2026"
)
