import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Streamlit UI
st.title("📰 Fake News Detection App")
st.write("Enter a news headline and the model will tell you if it's **real** or **fake**.")

# Text input
user_input = st.text_area("Enter News Headline:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a headline.")
    else:
        vectorized = vectorizer.transform([user_input])
        prediction = model.predict(vectorized)

        if prediction[0] == 1:
            st.success("✅ This appears to be **Real News**.")
        else:
            st.error("❌ This appears to be **Fake News**.")
