import streamlit as st
from predict import predict_ml, predict_lstm

st.set_page_config(page_title="AI Fake News Classifier", page_icon="🧠", layout="centered")

st.title("🧠 AI-Powered Fake News Classifier")
st.subheader("Detect whether a news article is Real or Fake")

text = st.text_area("Enter news text here:", height=200)

model_choice = st.radio("Choose Model Type:", ("Logistic Regression (ML)", "Random Forest (ML)", "LSTM (DL)"))

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        if model_choice == "Logistic Regression (ML)":
            result = predict_ml(text, model_type='logistic')
        elif model_choice == "Random Forest (ML)":
            result = predict_ml(text, model_type='rf')
        else:
            result = predict_lstm(text)

        if result == "Real":
            st.success("✅ This news article seems REAL.")
        else:
            st.error("❌ This news article seems FAKE.")
