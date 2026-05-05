import streamlit as st
import requests
import pickle

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Fake News Detection", layout="centered")

st.title("📰 Fake News Detection System")
st.write("B.Sc. Final Year Project | 2026")

# =========================
# LOAD CLASSICAL MODEL (SAFE)
# =========================
@st.cache_resource
def load_classical():
    try:
        vectorizer = pickle.load(open("saved_model/vectorizer.pkl", "rb"))
        scaler = pickle.load(open("saved_model/scaler.pkl", "rb"))
        model = pickle.load(open("saved_model/culture_features.pkl", "rb"))
        return vectorizer, scaler, model
    except:
        return None, None, None

vectorizer, scaler, rf_model = load_classical()

# =========================
# TOKEN
# =========================
HF_TOKEN = st.secrets["HF_TOKEN"]

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

# =========================
# HF API FUNCTION (COMMON)
# =========================
def hf_predict(repo_id, text):
    API_URL = f"https://api-inference.huggingface.co/models/{repo_id}"

    response = requests.post(
        API_URL,
        headers=headers,
        json={"inputs": text}
    )

    if response.status_code != 200:
        raise Exception(response.text)

    result = response.json()

    # safe extraction
    if isinstance(result, list):
        label = result[0]["label"]
    else:
        label = result["label"]

    # fix label issue
    if "REAL" in label.upper() or "LABEL_0" in label.upper():
        return 0
    else:
        return 1


# =========================
# INPUT
# =========================
text_input = st.text_area("Enter News Text")

model_choice = st.selectbox(
    "Select Model",
    ["Random Forest", "MuRIL", "mBERT", "XLM-RoBERTa"]
)

# =========================
# PREDICT BUTTON
# =========================
if st.button("Predict"):

    if text_input.strip() == "":
        st.warning("Please enter text")
    else:
        try:

            # =========================
            # RANDOM FOREST
            # =========================
            if model_choice == "Random Forest":
                if vectorizer is None:
                    st.error("Random Forest not available")
                    st.stop()

                vec = vectorizer.transform([text_input]).toarray()
                vec = scaler.transform(vec)
                pred = rf_model.predict(vec)[0]

            # =========================
            # MURIL (WORKING BASELINE)
            # =========================
            elif model_choice == "MuRIL":
                pred = hf_predict(
                    "google/muril-base-cased",
                    text_input
                )

            # =========================
            # MBERT (YOUR MODEL)
            # =========================
            elif model_choice == "mBERT":
                pred = hf_predict(
                    "riddhi04/mbert-hybrid-model",
                    text_input
                )

            # =========================
            # XLM-R (YOUR MODEL)
            # =========================
            elif model_choice == "XLM-RoBERTa":
                pred = hf_predict(
                    "riddhi04/xlmr-hybrid-model",
                    text_input
                )

            # =========================
            # OUTPUT
            # =========================
            if pred == 1:
                st.error("🚨 Fake News Detected")
            else:
                st.success("✅ Real News")

        except Exception as e:
            st.error("Error occurred during prediction")
            st.write(e)
