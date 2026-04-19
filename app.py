import streamlit as st
import joblib
import numpy as np
import os
import gdown
from datetime import datetime

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
# Theme toggle (UNCHANGED)
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
# Model download if missing
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
# Load components
# -------------------------------------------------

model = joblib.load("saved_model/model.pkl")
vectorizer = joblib.load("saved_model/vectorizer.pkl")
scaler = joblib.load("saved_model/scaler.pkl")
culture_features = joblib.load(
    "saved_model/culture_features.pkl"
)

# -------------------------------------------------
# Header (UNCHANGED)
# -------------------------------------------------

st.title("📰 Cross-Lingual Fake News Detection")

st.subheader(
    "AI-powered verification with Cultural Context Analysis"
)

st.markdown(
    "This system analyzes news text using machine learning "
    "to determine whether it is **Fake** or **Real**."
)

st.divider()

# -------------------------------------------------
# Sidebar (UNCHANGED)
# -------------------------------------------------

st.sidebar.title("About This System")

st.sidebar.write(
    """
• Machine Learning  
• Text Features  
• Cultural Context Analysis  
• Explainable AI (XAI)
"""
)

st.sidebar.write("Model: Random Forest")

# -------------------------------------------------
# Example buttons (UNCHANGED)
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
# Input box (UNCHANGED)
# -------------------------------------------------

user_text = st.text_area(
    "Enter News Text",
    height=180,
    value=st.session_state.user_text,
    placeholder="Paste news headline or article here..."
)

st.session_state.user_text = user_text

# -------------------------------------------------
# Cultural feature detection (FIXED ONLY HERE)
# -------------------------------------------------

def get_culture_features(text):

    text_lower = text.lower()

    features = {f: 0 for f in culture_features}

    detected = []

    # Keyword sets mapped directly to model features

    national_keywords = [
        "india", "indian", "nation",
        "country", "citizen", "government"
    ]

    religion_keywords = [
        "temple", "mosque", "church",
        "hindu", "muslim", "christian",
        "religion"
    ]

    community_keywords = [
        "riot", "clash", "violence",
        "tension", "conflict", "attack"
    ]

    health_keywords = [
        "vaccine", "virus", "covid",
        "disease", "cure", "medicine"
    ]

    festival_keywords = [
        "diwali", "eid",
        "christmas", "festival"
    ]

    political_keywords = [
        "election", "vote", "minister",
        "parliament", "party", "policy"
    ]

    regional_keywords = [
        "region", "border",
        "territory", "state"
    ]

    def count_keywords(keywords):
        return sum(1 for word in keywords if word in text_lower)

    # Counts

    if "national_identity_count" in features:
        count = count_keywords(national_keywords)
        features["national_identity_count"] = count
        if count > 0:
            detected.append("national_identity_count")

    if "religion_count" in features:
        count = count_keywords(religion_keywords)
        features["religion_count"] = count
        if count > 0:
            detected.append("religion_count")

    if "community_tension_count" in features:
        count = count_keywords(community_keywords)
        features["community_tension_count"] = count
        if count > 0:
            detected.append("community_tension_count")

    if "health_belief_count" in features:
        count = count_keywords(health_keywords)
        features["health_belief_count"] = count
        if count > 0:
            detected.append("health_belief_count")

    if "festival_count" in features:
        count = count_keywords(festival_keywords)
        features["festival_count"] = count
        if count > 0:
            detected.append("festival_count")

    if "political_symbol_count" in features:
        count = count_keywords(political_keywords)
        features["political_symbol_count"] = count
        if count > 0:
            detected.append("political_symbol_count")

    if "regional_identity_count" in features:
        count = count_keywords(regional_keywords)
        features["regional_identity_count"] = count
        if count > 0:
            detected.append("regional_identity_count")

    # Boolean flags

    for key in features:

        if key.startswith("has_"):

            base = key.replace("has_", "") + "_count"

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
# Prediction (UNCHANGED)
# -------------------------------------------------

if st.button("Predict"):

    if st.session_state.user_text.strip() == "":

        st.warning("Please enter news text.")

    else:

        try:

            with st.spinner(
                "Analyzing news content..."
            ):

                text_vector = vectorizer.transform(
                    [st.session_state.user_text]
                )

                culture_vector, detected_features = \
                    get_culture_features(
                        st.session_state.user_text
                    )

                if culture_vector.shape[1] == scaler.n_features_in_:
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

                confidence = round(
                    max(probability) * 100,
                    2
                )

                real_prob = round(
                    probability[0] * 100,
                    2
                )

                fake_prob = round(
                    probability[1] * 100,
                    2
                )

            if prediction == 1:

                st.error("🚨 Prediction: FAKE")

            else:

                st.success("✅ Prediction: REAL")

            st.write(
                f"Confidence: {confidence}%"
            )

            st.progress(confidence / 100)

            st.write(
                f"Fake Probability: {fake_prob}%"
            )

            st.write(
                f"Real Probability: {real_prob}%"
            )

            # Cultural features display

            st.subheader(
                "Detected Cultural Features"
            )

            if detected_features:

                for f in detected_features:
                    st.write("•", f)

            else:

                st.write(
                    "No cultural indicators detected."
                )

            # Top Influential Words (UNCHANGED)

            st.subheader(
                "Top Influential Words"
            )

            tfidf_vector = vectorizer.transform(
                [st.session_state.user_text]
            )

            feature_names = \
                vectorizer.get_feature_names_out()

            scores = tfidf_vector.toarray()[0]

            top_indices = scores.argsort()[::-1]

            top_words = []

            for idx in top_indices:

                word = feature_names[idx]

                if scores[idx] > 0:

                    if word not in [
                        "token_count",
                        "unique_token_count",
                        "lexical_diversity"
                    ]:

                        top_words.append(word)

                if len(top_words) == 5:
                    break

            if not top_words:

                nonzero_indices = np.where(scores > 0)[0]

                for idx in nonzero_indices[:5]:

                    top_words.append(
                        feature_names[idx]
                    )

            for word in top_words:

                st.write("•", word)

            if confidence < 60:

                st.warning(
                    "Low confidence prediction — result may be unreliable."
                )

            # Download report

            report = f"""
Fake News Detection Report

News Text:
{st.session_state.user_text}

Prediction:
{"FAKE" if prediction == 1 else "REAL"}

Confidence:
{confidence}%

Fake Probability:
{fake_prob}%

Real Probability:
{real_prob}%

Detected Cultural Features:
{', '.join(detected_features) if detected_features else 'None'}

Generated:
{datetime.now()}
"""

            st.download_button(

                label="Download Report",

                data=report,

                file_name="prediction_report.txt",

                mime="text/plain"

            )

        except Exception as e:

            st.error(
                "Error occurred during prediction."
            )

            st.write(str(e))

# -------------------------------------------------
# Footer (UNCHANGED)
# -------------------------------------------------

st.divider()

st.caption(
    "Cross-Lingual Fake News Detection with Cultural Context Analysis | 2026"
)
