import requests  # 🔥 MOVE THIS TO TOP (IMPORTANT)

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

                    culture_vector = get_culture_features(user_text)
                    culture_vector = scaler.transform(culture_vector)

                    combined = np.hstack(
                        (text_vector.toarray(), culture_vector)
                    )

                    prediction = int(model.predict(combined)[0])

                    probability = model.predict_proba(combined)[0]
                    confidence = probability[prediction]

                # ------------------------------
                # mBERT
                # ------------------------------
                elif model_choice == "mBERT":

                    token = st.secrets["HF_TOKEN_Riddhi"]

                    API_URL = "https://api-inference.huggingface.co/models/riddhi04/mbert-hybrid-model"

                    headers = {
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json"
                    }

                    response = requests.post(
                        API_URL,
                        headers=headers,
                        json={"inputs": user_text}
                    )

                    result = response.json()

                    if isinstance(result, list):
                        result = result[0]

                    label = result.get("label", "REAL")
                    score = result.get("score", 0.5)

                    prediction = 1 if "FAKE" in label.upper() else 0
                    confidence = score

                # ------------------------------
                # XLM-R
                # ------------------------------
                elif model_choice == "XLM-RoBERTa":

                    token = st.secrets["HF_TOKEN_Riddhi"]

                    API_URL = "https://api-inference.huggingface.co/models/riddhi04/xlmr-hybrid-model"

                    headers = {
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json"
                    }

                    response = requests.post(
                        API_URL,
                        headers=headers,
                        json={"inputs": user_text}
                    )

                    result = response.json()

                    if isinstance(result, list):
                        result = result[0]

                    label = result.get("label", "REAL")
                    score = result.get("score", 0.5)

                    prediction = 1 if "FAKE" in label.upper() else 0
                    confidence = score

                # ------------------------------
                # MuRIL (KEEP AS IS)
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
