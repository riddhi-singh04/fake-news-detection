import streamlit as st
import joblib
import numpy as np

# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

import os
import gdown

MODEL_PATH = "saved_model/model.pkl"

# Download model if not present
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
culture_features = joblib.load("saved_model/culture_features.pkl")

# -----------------------------
# Title
# -----------------------------

st.title("📰 Cross-Lingual Fake News Detection")

st.subheader(
    "with Cultural Context Analysis"
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
    This system detects fake news using:

    • Machine Learning  
    • Text Features  
    • Cultural Context Analysis  
    • Explainable AI (XAI)  
    """
)

st.sidebar.write("Model: Random Forest")

# -----------------------------
# Example buttons
# -----------------------------

st.subheader("Try Example News")

col1, col2 = st.columns(2)

example_text = ""

with col1:

    if st.button("Fake Example"):

        example_text = (
            "Scientists confirm drinking bleach cures all diseases instantly."
        )

with col2:

    if st.button("Real Example"):

        example_text = (
            "The government announced a new vaccination program for children."
        )

# -----------------------------
# User input
# -----------------------------

user_text = st.text_area(
    "Enter News Text",
    value=example_text,
    height=150
)

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

    if user_text.strip() == "":

        st.warning("Please enter news text.")

    else:

        try:

            with st.spinner("Analyzing news..."):

                # Text features

                text_vector = vectorizer.transform(
                    [user_text]
                )

                # Cultural features

                culture_vector, detected_features = \
                    get_culture_features(user_text)

                culture_vector = scaler.transform(
                    culture_vector
                )

                # Combine features

                combined = np.hstack(
                    (
                        text_vector.toarray(),
                        culture_vector
                    )
                )

                # Prediction

                prediction = int(
                    model.predict(combined)[0]
                )

                probability = model.predict_proba(
                    combined
                )[0]

                confidence = float(
                    round(max(probability) * 100, 2)
                )

                fake_prob = round(
                    probability[0] * 100,
                    2
                )

                real_prob = round(
                    probability[1] * 100,
                    2
                )

            # -----------------------------
            # Prediction result
            # -----------------------------

            if prediction == 1:

                st.success("✅ Prediction: REAL")

            else:

                st.error("🚨 Prediction: FAKE")

            st.write(f"Confidence: {confidence}%")

            st.progress(confidence / 100)

            # -----------------------------
            # Probability display
            # -----------------------------

            st.write(
                f"Fake Probability: {fake_prob}%"
            )

            st.write(
                f"Real Probability: {real_prob}%"
            )

            # -----------------------------
            # Cultural feature proof
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
            # XAI — Feature importance
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
            # Low confidence warning
            # -----------------------------

            if confidence < 60:

                st.warning(
                    "Low confidence prediction — result may be unreliable."
                )

        except Exception as e:

            st.error(
                "Error occurred during prediction."
            )

            st.write(str(e))

st.divider()

st.caption(
    "Cross-Lingual Fake News Detection with Cultural Context Analysis | 2026"
)