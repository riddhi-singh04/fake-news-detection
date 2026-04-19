import streamlit as st
import joblib
import numpy as np
import os
import gdown
from datetime import datetime

# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide"
)

# -----------------------------
# Session state
# -----------------------------

if "user_text" not in st.session_state:
    st.session_state.user_text = ""

# -----------------------------
# Theme toggle (working)
# -----------------------------

theme = st.sidebar.selectbox(
    "Choose Theme",
    ["Dark", "Light", "Smooth"]
)

if theme == "Dark":

    st.markdown("""
        <style>
        .stApp {
            background-color: #0e1117;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

elif theme == "Light":

    st.markdown("""
        <style>
        .stApp {
            background-color: #ffffff;
            color: black;
        }
        </style>
    """, unsafe_allow_html=True)

elif theme == "Smooth":

    st.markdown("""
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
    """, unsafe_allow_html=True)

# -----------------------------
# Model download if missing
# -----------------------------

MODEL_PATH = "saved_model/model.pkl"

if not os.path.exists(MODEL_PATH):

    os.makedirs("saved_model", exist_ok=True)

    gdown.download(
        "https://drive.google.com/uc?id=1uV2G8hRJF38FFABo-tWDPwtAxsDOmW5s",
        MODEL_PATH,
        quiet=False
    )

# -----------------------------
# Load components
# -----------------------------

model = joblib.load("saved_model/model.pkl")
vectorizer = joblib.load("saved_model/vectorizer.pkl")
scaler = joblib.load("saved_model/scaler.pkl")
culture_features = joblib.load(
    "saved_model/culture_features.pkl"
)

# -----------------------------
# Header
# -----------------------------

st.title("📰 Cross-Lingual Fake News Detection")

st.subheader(
    "AI-powered verification with Cultural Context Analysis"
)

st.markdown(
    "This system analyzes news text using machine learning "
    "to determine whether it is **Fake** or **Real**."
)

st.divider()

# -----------------------------
# Sidebar
# -----------------------------

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

# -----------------------------
# Example buttons (working)
# -----------------------------

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

# -----------------------------
# Input box
# -----------------------------

user_text = st.text_area(
    "Enter News Text",
    height=180,
    value=st.session_state.user_text,
    placeholder="Paste news headline or article here..."
)

st.session_state.user_text = user_text

# -----------------------------
# Cultural feature detection
# -----------------------------

def get_culture_features(text):

    text_lower = text.lower()

    features = []
    detected = []

    for feature in culture_features:

        if feature.lower() in text_lower:

            features.append(1)
            detected.append(feature)

        else:

            features.append(0)

    return np.array(features).reshape(1, -1), detected

# -----------------------------
# Prediction
# -----------------------------

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

                # CORRECT probability mapping
                real_prob = round(
                    probability[0] * 100,
                    2
                )

                fake_prob = round(
                    probability[1] * 100,
                    2
                )

            # -----------------------------
            # Correct label mapping
            # -----------------------------

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

            # -----------------------------
            # Cultural features
            # -----------------------------

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

            # -----------------------------
            # XAI
            # -----------------------------

            st.subheader(
                "Top Influential Features"
            )

            text_features = \
                vectorizer.get_feature_names_out()

            all_features = \
                list(text_features) + \
                list(culture_features)

            importances = \
                model.feature_importances_

            feature_importance = list(
                zip(all_features, importances)
            )

            top_features = sorted(
                feature_importance,
                key=lambda x: x[1],
                reverse=True
            )[:5]

            for feature, value in top_features:

                st.write("•", feature)

            # -----------------------------
            # Warning
            # -----------------------------

            if confidence < 60:

                st.warning(
                    "Low confidence prediction — result may be unreliable."
                )

            # -----------------------------
            # Download report
            # -----------------------------

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

# -----------------------------
# Footer
# -----------------------------

st.divider()

st.caption(
    "Cross-Lingual Fake News Detection with Cultural Context Analysis | 2026"
)
