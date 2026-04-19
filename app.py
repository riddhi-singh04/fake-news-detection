import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re

# ==============================
# Load Models
# ==============================

model = joblib.load("saved_model/model.pkl")
vectorizer = joblib.load("saved_model/vectorizer.pkl")
scaler = joblib.load("saved_model/scaler.pkl")
culture_features = joblib.load("saved_model/culture_features.pkl")

# ==============================
# Page Config
# ==============================

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="🧠",
    layout="wide"
)

# ==============================
# Theme Styling
# ==============================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
    color: white;
}

.stButton>button {
    border-radius: 8px;
    padding: 10px 20px;
}

</style>
""", unsafe_allow_html=True)

# ==============================
# Title
# ==============================

st.title("AI-Powered Fake News Detection with Cultural Context Analysis")

st.write(
"This system analyzes news text using machine learning and cultural indicators."
)

# ==============================
# Feature Labels
# ==============================

FEATURE_LABELS = {

    "religion_count": "Religious references",
    "political_symbol_count": "Political symbols",
    "regional_identity_count": "Regional identity references",
    "community_tension_count": "Community tension indicators",
    "health_belief_count": "Health belief claims",
    "festival_count": "Festival references",
    "national_identity_count": "National identity references",

    "has_religion": "Religion mentioned",
    "has_political_symbol": "Political symbol mentioned",
    "has_regional_identity": "Regional identity mentioned",
    "has_community_tension": "Community tension mentioned",
    "has_health_belief": "Health belief mentioned",
    "has_festival": "Festival mentioned",
    "has_national_identity": "Nation/Country mentioned"

}

# ==============================
# Keyword Dictionary
# ==============================

FEATURE_KEYWORDS = {

    "religion": [
        "temple","mosque","church","hindu","muslim","christian",
        "religion","faith"
    ],

    "political": [
        "election","vote","minister","government",
        "parliament","party","policy"
    ],

    "regional": [
        "region","border","territory","state"
    ],

    "community": [
        "riot","clash","violence","tension",
        "conflict","attack"
    ],

    "health": [
        "vaccine","virus","covid",
        "disease","cure","medicine"
    ],

    "festival": [
        "diwali","eid","christmas","festival"
    ],

    "national": [
        "india","indian","nation",
        "country","citizen"
    ]

}

# ==============================
# Text Cleaning
# ==============================

def clean_text(text):

    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    return text


# ==============================
# Cultural Feature Detection
# ==============================

def detect_cultural_features(text):

    text = text.lower()

    features = {f: 0 for f in culture_features}

    def count_keywords(keywords):
        return sum(1 for word in keywords if word in text)

    if "religion_count" in features:
        features["religion_count"] = count_keywords(
            FEATURE_KEYWORDS["religion"]
        )

    if "political_symbol_count" in features:
        features["political_symbol_count"] = count_keywords(
            FEATURE_KEYWORDS["political"]
        )

    if "regional_identity_count" in features:
        features["regional_identity_count"] = count_keywords(
            FEATURE_KEYWORDS["regional"]
        )

    if "community_tension_count" in features:
        features["community_tension_count"] = count_keywords(
            FEATURE_KEYWORDS["community"]
        )

    if "health_belief_count" in features:
        features["health_belief_count"] = count_keywords(
            FEATURE_KEYWORDS["health"]
        )

    if "festival_count" in features:
        features["festival_count"] = count_keywords(
            FEATURE_KEYWORDS["festival"]
        )

    if "national_identity_count" in features:
        features["national_identity_count"] = count_keywords(
            FEATURE_KEYWORDS["national"]
        )

    # Boolean indicators

    if "has_religion" in features:
        features["has_religion"] = int(
            features.get("religion_count",0) > 0
        )

    if "has_political_symbol" in features:
        features["has_political_symbol"] = int(
            features.get("political_symbol_count",0) > 0
        )

    if "has_regional_identity" in features:
        features["has_regional_identity"] = int(
            features.get("regional_identity_count",0) > 0
        )

    if "has_community_tension" in features:
        features["has_community_tension"] = int(
            features.get("community_tension_count",0) > 0
        )

    if "has_health_belief" in features:
        features["has_health_belief"] = int(
            features.get("health_belief_count",0) > 0
        )

    if "has_festival" in features:
        features["has_festival"] = int(
            features.get("festival_count",0) > 0
        )

    if "has_national_identity" in features:
        features["has_national_identity"] = int(
            features.get("national_identity_count",0) > 0
        )

    return features


# ==============================
# Top Influential Words
# ==============================

def get_top_words(vectorizer, vector, top_n=5):

    feature_names = vectorizer.get_feature_names_out()

    sorted_indices = vector.toarray()[0].argsort()[::-1]

    top_words = [
        feature_names[i]
        for i in sorted_indices[:top_n]
        if vector.toarray()[0][i] > 0
    ]

    return top_words


# ==============================
# UI Input
# ==============================

st.header("Enter News Text")

news_text = st.text_area(
    "Paste news headline or article",
    height=200
)

# ==============================
# Prediction
# ==============================

if st.button("Predict"):

    if news_text.strip() == "":

        st.warning("Please enter news text.")

    else:

        try:

            clean = clean_text(news_text)

            text_vector = vectorizer.transform([clean])

            cultural_dict = detect_cultural_features(clean)

            cultural_df = pd.DataFrame(
                [cultural_dict]
            )

            cultural_scaled = scaler.transform(
                cultural_df
            )

            final_features = np.hstack(
                (
                    text_vector.toarray(),
                    cultural_scaled
                )
            )

            prediction = model.predict(
                final_features
            )[0]

            probs = model.predict_proba(
                final_features
            )[0]

            # Mapping
            label = "FAKE" if prediction == 1 else "REAL"

            confidence = max(probs)

            if label == "REAL":

                st.success(
                    f"Prediction: {label}"
                )

            else:

                st.error(
                    f"Prediction: {label}"
                )

            st.write(
                f"Confidence: {confidence:.2%}"
            )

            st.progress(
                float(confidence)
            )

            st.write(
                f"Fake Probability: {probs[1]:.2%}"
            )

            st.write(
                f"Real Probability: {probs[0]:.2%}"
            )

            # ==========================
            # Cultural Features Display
            # ==========================

            st.header(
                "Detected Cultural Features"
            )

            detected = [
                f for f,v in cultural_dict.items()
                if v > 0
            ]

            if detected:

                for f in detected:

                    label_name = FEATURE_LABELS.get(
                        f,
                        f
                    )

                    st.write(
                        f"• {label_name}"
                    )

            else:

                st.write(
                    "No cultural indicators detected."
                )

            # ==========================
            # Influential Words
            # ==========================

            st.header(
                "Top Influential Words"
            )

            top_words = get_top_words(
                vectorizer,
                text_vector
            )

            for word in top_words:

                st.write(
                    f"• {word}"
                )

            if confidence < 0.6:

                st.warning(
                    "Low confidence prediction — result may be unreliable."
                )

        except Exception as e:

            st.error(
                "Error occurred during prediction."
            )

            st.write(str(e))
