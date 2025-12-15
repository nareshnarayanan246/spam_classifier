import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.set_page_config(page_title="Spam Classifier", page_icon="📧")

st.title("📧 Spam Classifier Demo")
st.write("AI Python Capstone Project")

message = st.text_area("Enter your message:")

if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        message_vec = vectorizer.transform([message])
        prediction = model.predict(message_vec)[0]

        if prediction == 1:
            st.error("🚨 This message is SPAM")
        else:
            st.success("✅ This message is NOT SPAM")
