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
    page_title="Cross-Lingual Fake News Detection",
    page_icon="📰",
    layout="wide"
)

# -------------------------------------------------
# Custom CSS
# -------------------------------------------------

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    .stApp { font-family: 'IBM Plex Sans', sans-serif; }

    .cultural-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border: 1px solid #e94560;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        color: white;
    }

    .xai-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 13px;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #e94560;
        margin-bottom: 12px;
    }

    .feature-bar-container {
        margin: 8px 0;
    }

    .feature-label {
        font-size: 13px;
        color: #a8b2d8;
        margin-bottom: 3px;
        display: flex;
        justify-content: space-between;
    }

    .keyword-tag {
        display: inline-block;
        background: rgba(233, 69, 96, 0.2);
        border: 1px solid #e94560;
        border-radius: 20px;
        padding: 3px 12px;
        margin: 3px;
        font-size: 12px;
        color: #e94560;
        font-family: 'IBM Plex Mono', monospace;
    }

    .keyword-tag-found {
        background: rgba(100, 255, 150, 0.15);
        border-color: #64ff96;
        color: #64ff96;
    }

    .verdict-fake {
        background: linear-gradient(135deg, #3d0000, #6b0000);
        border: 2px solid #ff4444;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        color: white;
    }

    .verdict-real {
        background: linear-gradient(135deg, #003d00, #006b00);
        border: 2px solid #44ff44;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        color: white;
    }

    .verdict-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 28px;
        font-weight: 600;
        letter-spacing: 3px;
    }

    .confidence-text {
        font-size: 14px;
        opacity: 0.8;
        margin-top: 8px;
    }

    .model-badge {
        display: inline-block;
        background: rgba(233, 69, 96, 0.15);
        border: 1px solid #e94560;
        border-radius: 6px;
        padding: 2px 10px;
        font-size: 11px;
        font-family: 'IBM Plex Mono', monospace;
        color: #e94560;
        letter-spacing: 1px;
    }

    .section-divider {
        border: none;
        border-top: 1px solid rgba(233, 69, 96, 0.3);
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Session state
# -------------------------------------------------

if "user_text" not in st.session_state:
    st.session_state.user_text = ""

# -------------------------------------------------
# Theme toggle
# -------------------------------------------------

theme = st.sidebar.selectbox("Choose Theme", ["Dark", "Light", "Smooth"])

if theme == "Dark":
    st.markdown("<style>.stApp {background-color: #0e1117; color: white;}</style>", unsafe_allow_html=True)
elif theme == "Light":
    st.markdown("<style>.stApp {background-color: #f5f5f5; color: #111;}</style>", unsafe_allow_html=True)
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

st.markdown("## 📰 Cross-Lingual Fake News Detection")
st.markdown("**Cultural Feature Analysis & Explainable AI**")
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
    if st.button("📄 Load Real News Example"):
        st.session_state.user_text = "The Reserve Bank of India announced a revision in repo rates to control inflation."

with col2:
    if st.button("⚠️ Load Fake News Example"):
        st.session_state.user_text = "Scientists confirm drinking bleach cures all diseases instantly."

# -------------------------------------------------
# Input
# -------------------------------------------------

user_text = st.text_area("Enter News Text", height=180, value=st.session_state.user_text)
st.session_state.user_text = user_text

# -------------------------------------------------
# Cultural Feature Extraction (enhanced)
# -------------------------------------------------

# Full keyword map with categories and emojis for display
CULTURAL_KEYWORD_MAP = {
    "national_identity_count": {
        "label": "National Identity",
        "emoji": "🏛️",
        "keywords": ["india", "country", "nation", "government", "national", "state", "republic", "ministry", "parliament", "prime minister", "modi", "delhi"],
        "description": "References to national identity, politics, and governance"
    },
    "health_belief_count": {
        "label": "Health & Medicine",
        "emoji": "🏥",
        "keywords": ["cure", "disease", "virus", "vaccine", "medicine", "hospital", "doctor", "health", "covid", "ayurveda", "treatment", "remedy"],
        "description": "Health misinformation is a major vector in Indian fake news"
    },
    "community_tension_count": {
        "label": "Community Tension",
        "emoji": "⚡",
        "keywords": ["riot", "violence", "attack", "communal", "protest", "clash", "mob", "tension", "unrest", "conflict", "religion", "caste"],
        "description": "Communal and social tension indicators"
    },
    "education_count": {
        "label": "Education & Authority",
        "emoji": "🎓",
        "keywords": ["school", "college", "university", "research", "study", "scientist", "expert", "professor", "iit", "report", "study", "survey"],
        "description": "Authority and credibility signals in news"
    }
}

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


def get_detailed_cultural_analysis(text):
    """Returns detailed cultural feature analysis for XAI display."""
    text_lower = text.lower()
    analysis = {}

    for feature_key, feature_info in CULTURAL_KEYWORD_MAP.items():
        found_keywords = [kw for kw in feature_info["keywords"] if kw in text_lower]
        count = len(found_keywords)
        analysis[feature_key] = {
            "label": feature_info["label"],
            "emoji": feature_info["emoji"],
            "count": count,
            "found_keywords": found_keywords,
            "all_keywords": feature_info["keywords"],
            "description": feature_info["description"],
            "score": min(count / 3.0, 1.0)  # normalize to 0-1
        }

    return analysis


def render_cultural_analysis(text, model_name, rf_importances=None):
    """Renders the full XAI / cultural features panel."""

    analysis = get_detailed_cultural_analysis(text)
    total_cultural_signals = sum(v["count"] for v in analysis.values())

    st.markdown("---")
    st.markdown("### 🔍 Explainability & Cultural Feature Analysis")
    st.markdown(
        f"<span class='model-badge'>{model_name}</span> &nbsp; "
        f"**{total_cultural_signals}** cultural signal(s) detected in this text",
        unsafe_allow_html=True
    )

    # ── Two columns: cultural features | keyword breakdown ──
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("#### 📊 Cultural Feature Scores")
        st.caption("How strongly each cultural category is represented in the text")

        for key, data in analysis.items():
            score_pct = int(data["score"] * 100)
            color = "#64ff96" if data["count"] > 0 else "#555577"

            st.markdown(f"""
            <div class='feature-bar-container'>
                <div class='feature-label'>
                    <span>{data['emoji']} {data['label']}</span>
                    <span style='color:{color}; font-weight:600'>{data['count']} signal(s)</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.progress(data["score"], text=f"{score_pct}% activation")

    with col_right:
        st.markdown("#### 🏷️ Keyword Detection")
        st.caption("Keywords found in text that trigger each cultural category")

        for key, data in analysis.items():
            with st.expander(f"{data['emoji']} {data['label']} — {data['count']} found", expanded=data["count"] > 0):
                st.caption(data["description"])

                if data["found_keywords"]:
                    found_html = " ".join([
                        f"<span class='keyword-tag keyword-tag-found'>✓ {kw}</span>"
                        for kw in data["found_keywords"]
                    ])
                    st.markdown(f"**Detected:** {found_html}", unsafe_allow_html=True)
                else:
                    st.markdown("*No keywords from this category detected*")

    # ── RandomForest specific: feature importances ──
    if rf_importances is not None:
        st.markdown("---")
        st.markdown("#### 🌲 RandomForest — Top Influential Features")
        st.caption("The most important features this model used to make its decision")

        # Get top N TF-IDF features
        feature_names = vectorizer.get_feature_names_out()
        cultural_feat_names = list(culture_features)
        all_feature_names = list(feature_names) + cultural_feat_names

        importances = rf_importances
        top_indices = np.argsort(importances)[::-1][:15]

        top_features = []
        for idx in top_indices:
            if idx < len(all_feature_names):
                top_features.append({
                    "name": all_feature_names[idx],
                    "importance": float(importances[idx]),
                    "is_cultural": idx >= len(feature_names)
                })

        if top_features:
            max_imp = top_features[0]["importance"] if top_features else 1

            for feat in top_features:
                normalized = feat["importance"] / max_imp if max_imp > 0 else 0
                tag = "🏛️ Cultural" if feat["is_cultural"] else "📝 Text"
                color = "#e94560" if feat["is_cultural"] else "#4a9eff"

                col_name, col_bar, col_val = st.columns([3, 5, 1])
                with col_name:
                    st.markdown(
                        f"<span style='color:{color}; font-size:12px'>{tag}</span><br>"
                        f"<code>{feat['name']}</code>",
                        unsafe_allow_html=True
                    )
                with col_bar:
                    st.progress(normalized)
                with col_val:
                    st.markdown(
                        f"<span style='font-size:11px; color:#888'>{feat['importance']:.4f}</span>",
                        unsafe_allow_html=True
                    )

    # ── Cultural risk summary ──
    st.markdown("---")
    st.markdown("#### 📋 Cultural Context Summary")

    dominant = max(analysis.values(), key=lambda x: x["count"])
    if dominant["count"] > 0:
        st.info(
            f"**Dominant signal:** {dominant['emoji']} {dominant['label']} "
            f"({dominant['count']} keyword(s): {', '.join(dominant['found_keywords'])})\n\n"
            f"{dominant['description']}"
        )
    else:
        st.info(
            "No strong cultural signals detected. "
            "The prediction is based primarily on linguistic/semantic patterns."
        )

    # ── Cross-lingual note ──
    with st.expander("ℹ️ About Cross-Lingual Cultural Analysis"):
        st.markdown("""
        This system uses **cultural feature augmentation** to improve fake news detection
        across languages. Standard models trained on English data often miss culturally
        specific misinformation patterns in Hindi, Hinglish, or regional Indian news.

        The cultural features capture:
        - **National Identity**: Politically charged language common in Indian fake news
        - **Health Beliefs**: Ayurvedic/alternative medicine misinformation
        - **Community Tension**: Communal harmony threats — a major fake news category in India
        - **Education/Authority**: Fake expert citations and false research claims

        These features are combined with **multilingual BERT embeddings** (mBERT, XLM-R, MuRIL)
        that understand language-agnostic semantic meaning, enabling cross-lingual detection.
        """)


# -------------------------------------------------
# Prediction
# -------------------------------------------------

if st.button("🔍 Predict", type="primary"):

    if user_text.strip() == "":
        st.warning("Please enter news text.")

    else:
        try:
            with st.spinner("Analyzing news content..."):

                rf_importances = None  # only set for RandomForest

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

                    # Get feature importances for XAI
                    if hasattr(model, "feature_importances_"):
                        rf_importances = model.feature_importances_

                # ------------------------------
                # mBERT
                # ------------------------------
                elif model_choice == "mBERT":
                    token = st.secrets["HF_TOKEN_Riddhi"]

                    client = Client(
                        "riddhi04/mbert-hybrid-model",
                        token=token
                    )

                    result = client.predict(
                        user_text,
                        api_name="/predict"
                    )

                    result_str = str(result)
                    prediction = 1 if "FAKE" in result_str.upper() else 0

                    match = re.search(r"(\d+(\.\d+)?)%", result_str)
                    confidence = float(match.group(1)) / 100 if match else 0.85

                # ------------------------------
                # XLM-RoBERTa
                # ------------------------------
                elif model_choice == "XLM-RoBERTa":
                    token = st.secrets["HF_TOKEN_Riddhi"]

                    client = Client(
                        "riddhi04/xlmr-hybrid-model",
                        token=token
                    )

                    result = client.predict(
                        user_text,
                        api_name="/predict"
                    )

                    result_str = str(result)
                    prediction = 1 if "FAKE" in result_str.upper() else 0

                    match = re.search(r"(\d+(\.\d+)?)%", result_str)
                    confidence = float(match.group(1)) / 100 if match else 0.85

                # ------------------------------
                # MuRIL
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

            # -------------------------------------------------
            # Result display
            # -------------------------------------------------

            st.markdown("---")
            st.markdown("### 🎯 Prediction Result")

            col_result, col_conf = st.columns([2, 1])

            with col_result:
                if prediction == 1:
                    st.markdown("""
                    <div class='verdict-fake'>
                        <div class='verdict-title'>🚨 FAKE</div>
                        <div class='confidence-text'>This article shows signs of misinformation</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class='verdict-real'>
                        <div class='verdict-title'>✅ REAL</div>
                        <div class='confidence-text'>This article appears to be credible</div>
                    </div>
                    """, unsafe_allow_html=True)

            with col_conf:
                st.metric(
                    label="Model Confidence",
                    value=f"{round(confidence * 100, 2)}%",
                    delta=f"{'High' if confidence > 0.8 else 'Medium' if confidence > 0.6 else 'Low'} confidence"
                )
                st.caption(f"Model: **{model_choice}**")

            # -------------------------------------------------
            # XAI + Cultural Features Panel
            # -------------------------------------------------

            render_cultural_analysis(
                text=user_text,
                model_name=model_choice,
                rf_importances=rf_importances
            )

        except Exception as e:
            st.error("Error occurred during prediction.")
            st.write(str(e))

st.divider()
st.caption("B.Sc. Final Year Project | 2026 | Cross-Lingual Fake News Detection with Cultural Feature Analysis")
