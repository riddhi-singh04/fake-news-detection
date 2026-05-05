import streamlit as st
import numpy as np
import joblib
import re
from gradio_client import Client

# ---------------- UI ----------------

st.title("Fake News Detection")

user_text = st.text_area("Enter news text")

model_choice = st.selectbox(
    "Select Model",
    ["RandomForest", "mBERT", "XLM-RoBERTa", "MuRIL"]
)

# ---------------- LOAD MODELS ----------------

@st.cache_resource
def load_models():
    try:
        vectorizer = joblib.load("saved_model/vectorizer.pkl")
        scaler = joblib.load("saved_model/scaler.pkl")
        model = joblib.load("saved_model/culture_features.pkl")
        return vectorizer, scaler, model
    except:
        return None, None, None

vectorizer, scaler, model = load_models()

# Dummy function (your existing one should already be there)
def get_culture_features(text):
    return np.zeros((1, 10))  # keep as your original if already defined


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
                # RANDOM FOREST (LOCAL)
                # ------------------------------
                if model_choice == "RandomForest":

                    if vectorizer is None:
                        st.error("Model not loaded")
                        st.stop()

                    text_vector = vectorizer.transform([user_text])

                    culture_vector = get_culture_features(user_text)
                    culture_vector = scaler.transform(culture_vector)

                    combined = np.hstack(
                        (text_vector.toarray(), culture_vector)
                    )

                    prediction = int(model.predict(combined)[0])

                    probability = model.predict_proba(combined)[0]
                    confidence = probability[prediction]

                # ------------------------------
                # mBERT (SAME AS MURIL STYLE)
                # ------------------------------
                elif model_choice == "mBERT":

                    token = st.secrets["HF_TOKEN_Riddhi"]

                    client = Client(
                        "riddhi04/mbert-hybrid-model",
                        hf_token=token
                    )

                    result = client.predict(
                        text=user_text,
                        api_name="/predict"
                    )

                    result_str = str(result)

                    prediction = 1 if "FAKE" in result_str.upper() else 0

                    match = re.search(r"(\d+(\.\d+)?)%", result_str)
                    confidence = (
                        float(match.group(1)) / 100 if match else 0.85
                    )

                # ------------------------------
                # XLM-R (SAME AS MURIL STYLE)
                # ------------------------------
                elif model_choice == "XLM-RoBERTa":

                    token = st.secrets["HF_TOKEN_Riddhi"]

                    client = Client(
                        "riddhi04/xlmr-hybrid-model",
                        hf_token=token
                    )

                    result = client.predict(
                        text=user_text,
                        api_name="/predict"
                    )

                    result_str = str(result)

                    prediction = 1 if "FAKE" in result_str.upper() else 0

                    match = re.search(r"(\d+(\.\d+)?)%", result_str)
                    confidence = (
                        float(match.group(1)) / 100 if match else 0.85
                    )

                # ------------------------------
                # MuRIL (UNCHANGED)
                # ------------------------------
                elif model_choice == "MuRIL":

                    token = st.secrets["HF_TOKEN"]

                    client = Client(
                        "pseudokoo/FakeNews-Detector-1-API",
                        hf_token=token
                    )

                    result = client.predict(
                        text=user_text,
                        api_name="/predict_fake_news"
                    )

                    result_str = str(result)

                    prediction = 1 if "FAKE" in result_str.upper() else 0

                    match = re.search(r"(\d+(\.\d+)?)%", result_str)
                    confidence = (
                        float(match.group(1)) / 100 if match else 0.85
                    )

            # ------------------------------
            # OUTPUT
            # ------------------------------

            if prediction == 1:
                st.error("🚨 Prediction: FAKE")
            else:
                st.success("✅ Prediction: REAL")

            st.write(f"Confidence: {round(confidence*100,2)}%")

        except Exception as e:
            st.error("Error occurred during prediction.")
            st.write(str(e))
