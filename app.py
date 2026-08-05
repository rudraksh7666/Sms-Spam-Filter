import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# --- One-time setup ---
nltk.download("stopwords")
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


def clean_text(text):
    """
    IMPORTANT: this must match the cleaning function you used in your notebook
    to build `message_clean` — copy your exact version here, otherwise the
    model will see differently-processed text than it was trained on.
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)


# --- Load model (cached so it only loads once) ---
@st.cache_resource
def load_model():
    vectorizer = joblib.load("spam_vectorizer.pkl")
    model = joblib.load("spam_model.pkl")
    return vectorizer, model


vectorizer, model = load_model()

# --- UI ---
st.set_page_config(page_title="SMS Spam Filter", page_icon="📱")
st.title("📱 SMS Spam Filter")
st.write(
    "Classifies a message as **spam** or **ham** using a Multinomial Naive Bayes "
    "model — the classical ML baseline from a three-way comparison "
    "(Naive Bayes vs. GloVe+LSTM vs. fine-tuned BERT)."
)

message = st.text_area("Enter an SMS message:", height=100, placeholder="e.g. Congratulations! You've won a free prize, claim now!")

if st.button("Check Message", type="primary"):
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        cleaned = clean_text(message)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]

        if prediction == 1:  # adjust if your label encoding is flipped
            st.error(f"🚨 **SPAM** (confidence: {proba.max():.1%})")
        else:
            st.success(f"✅ **HAM** — looks legitimate (confidence: {proba.max():.1%})")

st.caption("Full project with the LSTM and BERT comparison: [GitHub repo](https://github.com/rudraksh7666/Sms-Spam-Filter)")
