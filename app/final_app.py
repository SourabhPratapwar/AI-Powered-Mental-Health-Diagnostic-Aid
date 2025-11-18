import os, re
import numpy as np
import pandas as pd
import streamlit as st
import torch
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from datetime import datetime

# ------------------ Hugging Face ------------------
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ------------------ NLTK ------------------
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

def _ensure_nltk():
    pkgs = ["punkt", "stopwords", "vader_lexicon"]
    for p in pkgs:
        try:
            nltk.data.find(f"tokenizers/{p}" if p == "punkt" else f"corpora/{p}")
        except LookupError:
            nltk.download(p, quiet=True)

_ensure_nltk()

st.set_page_config(
    page_title="🧠 Mental Health Analyzer",
    layout="centered",
    initial_sidebar_state="auto"
)

# ------------------ Constants ------------------
LOG_FILE = "patient_analysis_log.csv"
EMOTION_LABELS = ["Stress", "Depression", "Bipolar", "Personality disorder", "Anxiety"]

# ------------------ Preprocessing ------------------
def clean_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    sw = set(stopwords.words("english"))
    return " ".join([w for w in tokens if w not in sw])

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv("dataset/data_to_be_cleansed.csv")
    # be robust if 'title' is not present
    title_col = df["title"].fillna("") if "title" in df.columns else ""
    text_col  = df["text"].fillna("")  if "text"  in df.columns else ""
    df["full_text"] = (title_col.astype(str) + ". " + text_col.astype(str)).str.strip()
    df["cleaned_text"] = df["full_text"].apply(clean_text)
    return df

@st.cache_data(show_spinner=False)
def init_vectorizer(dataframe):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words="english", max_features=100)
    vectorizer.fit(dataframe["cleaned_text"])
    return vectorizer

# ------------------ Emotion Detection (keyword rules) ------------------
EMOTION_KEYWORDS = {
    "sadness": ["sad", "depressed", "cry", "alone", "worthless"],
    "fear": ["afraid", "scared", "anxious", "panic"],
    "anger": ["angry", "mad", "hate", "frustrated"],
    "joy": ["happy", "joy", "glad", "grateful"],
    "love": ["love", "loved", "hug", "friend"],
}
sia = SentimentIntensityAnalyzer()

def detect_emotions(text: str):
    detected = set()
    for emotion, words in EMOTION_KEYWORDS.items():
        if any(f" {w} " in f" {text} " for w in words):
            detected.add(emotion)
    return list(detected) if detected else ["neutral"]

def extract_keywords(text: str, vectorizer):
    tfidf = vectorizer.transform([text])
    scores = tfidf.toarray().flatten()
    feats = vectorizer.get_feature_names_out()
    top_idx = scores.argsort()[::-1][:5]
    return [feats[i] for i in top_idx if scores[i] > 0]

def generate_summary(text: str, vectorizer):
    cleaned = clean_text(text)
    score = sia.polarity_scores(cleaned)["compound"]
    sentiment = "POSITIVE" if score >= 0.05 else "NEGATIVE" if score <= -0.05 else "NEUTRAL"
    emotions = detect_emotions(cleaned)
    keywords = extract_keywords(cleaned, vectorizer)
    evaluation = []
    if "sadness" in emotions or sentiment == "NEGATIVE":
        evaluation.append("⚠️ Possible depression risk")
    if "fear" in emotions:
        evaluation.append("⚠️ Anxiety indicators")
    if "anger" in emotions:
        evaluation.append("⚠️ Trauma or irritability")
    if any(e in emotions for e in ["joy", "love"]):
        evaluation.append("✅ Positive signs")
    if not evaluation:
        evaluation.append("📝 General check-in recommended")
    return {
        "Sentiment": sentiment,
        "Emotions": emotions,
        "Keywords": keywords,
        "Evaluation": evaluation,
        "Cleaned": cleaned,
    }

# ------------------ Load fine-tuned BERT ------------------
@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer(model_dir="./emotion_model"):
    # device selection (CUDA -> MPS -> CPU)
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
  
    if getattr(model.config, "problem_type", None) != "multi_label_classification":
        model.config.problem_type = "multi_label_classification"
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    return model, tokenizer, device

def analyze_model(text: str, model, tokenizer, device):
    with torch.no_grad():
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).squeeze(0).detach().cpu().numpy()
    return (probs * 100).round(2)

# ------------------ Save Logs ------------------
def save_entry(pid, text, report, ml_scores):
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "patient_id": pid,
        "text": text,
        "sentiment": report["Sentiment"],
        "emotions": ",".join(report["Emotions"]),
        "keywords": ",".join(report["Keywords"]),
        "evaluation": ",".join(report["Evaluation"]),
    }
    row.update({f"BERT_{lbl}": val for lbl, val in zip(EMOTION_LABELS, ml_scores)})
    df = pd.DataFrame([row])
    df.to_csv(LOG_FILE, mode="a", header=not os.path.exists(LOG_FILE), index=False)

# ------------------ Visuals ------------------
def plot_radar(labels, values):
    values = np.array(values, dtype=float)
    stats = np.concatenate([values, values[:1]])
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate([angles, angles[:1]])

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, stats, "o-", linewidth=2)
    ax.fill(angles, stats, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title("Emotion Radar")
    st.pyplot(fig)

def plot_wordcloud(text):
    wc = WordCloud(width=600, height=400, background_color="white").generate(text)
    st.image(wc.to_array(), caption="☁️ Word Cloud", use_container_width=True)

# ------------------ UI ------------------
st.title("🧠 Mental Health Text Analyzer")
st.caption("Explore emotional and sentiment patterns from patient text.")

pid = st.text_input("🆔 Patient ID", placeholder="e.g. P100")
text_input = st.text_area("📝 Patient Description", height=180, placeholder="e.g. I feel anxious and can't sleep...")

if not text_input:
    st.info("Please enter patient text above to begin analysis.")
else:
    data_df = load_data()
    vectorizer = init_vectorizer(data_df)
    model, tokenizer, device = load_model_and_tokenizer("./emotion_model")

    tab1, tab2, tab3 = st.tabs(["📋 Rule-Based", "🤖 ML-Based", "📊 Visualizations"])

    with tab1:
        st.subheader("📋 Sentiment & Emotion Summary")
        result = generate_summary(text_input, vectorizer)
        st.markdown(f"**Sentiment:** `{result['Sentiment']}`")
        st.markdown(f"**Emotions:** `{', '.join(result['Emotions'])}`")
        st.markdown(f"**Top Keywords:** `{', '.join(result['Keywords'])}`")
        st.markdown("**Suggested Evaluation:**")
        for item in result["Evaluation"]:
            st.write(f"- {item}")
        plot_wordcloud(result["Cleaned"])

    with tab2:
        st.subheader("🤖 BERT-Based Emotion Detection")
        ml_scores = analyze_model(text_input, model, tokenizer, device)
        for label, score in zip(EMOTION_LABELS, ml_scores):
            st.markdown(f"**{label}**: {score:.2f}%")
            st.progress(float(score) / 100.0)
        if st.button("💾 Save Analysis", use_container_width=True):
            if pid.strip():
                save_entry(pid.strip(), text_input, result, ml_scores)
                st.success(f"Saved analysis for Patient {pid.strip()}")
            else:
                st.warning("Enter Patient ID to save")

    with tab3:
        st.subheader("📊 Radar Emotion Chart")
        plot_radar(EMOTION_LABELS, ml_scores if "ml_scores" in locals() else [0] * len(EMOTION_LABELS))
