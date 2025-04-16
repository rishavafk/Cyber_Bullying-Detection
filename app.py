import streamlit as st
import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Download stopwords once
nltk.download('stopwords')

# --- Preprocessing Function ---
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# --- Load Model & Vectorizer ---
@st.cache_resource
def load_model_and_vectorizer():
    with open("bullying_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


# --- UI ---
st.title("🚨 Cyberbullying Tweet Classifier")
st.markdown("This app uses a machine learning model to detect cyberbullying in tweets. Enter a tweet below to classify:")

tweet_input = st.text_area("📥 Enter a tweet for analysis:")

if st.button("🚀 Classify Tweet"):
    if tweet_input:
        try:
            model, vectorizer = load_model_and_vectorizer()
            cleaned_text = preprocess_text(tweet_input)
            transformed_input = vectorizer.transform([cleaned_text])
            prediction = model.predict(transformed_input)[0]

            # Updated classification logic
            if prediction == 1:
                result = "⚠️ This is cyberbullying!"
            else:
                result = "🚫 This can be Cyberbullying"
            
            st.success(f"🧠 Classification Result: **{result}**")

        except FileNotFoundError:
            st.error("Model/vectorizer files missing")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter a tweet")

# Optional: Show sample data


if st.checkbox("📄 Show sample tweets from dataset"):
    try:
        df = pd.read_csv("cyberbullying_tweets.csv")
        st.dataframe(df.head(10).style.applymap(
            lambda x: 'background-color: #ffcccc' if x != 'not_cyberbullying' else '', 
            subset=['cyberbullying_type']
        ))
    except FileNotFoundError:
        st.error("CSV file not found. Please ensure 'cyberbullying_tweets.csv' is in the app directory.")
