# app.py
import streamlit as st

import os
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

st.set_page_config(page_title="Tweet Sentiment (TFIDF / BERT)", layout="wide")

st.title("🧠 Tweet Sentiment Analyzer")
st.markdown("Choose model: Baseline (TF-IDF + Logistic) or Transformer (BERT)")

# paths
WORK_DIR = "models"   # ضع مجلد النماذج هنا أو اسم المجلد الذي حملت إليه الملفات
baseline_path = os.path.join(WORK_DIR, "baseline_tfidf_logreg.joblib")
bert_dir = os.path.join(WORK_DIR, "bert_sentiment_model")  # إن حفظت fine-tuned model هنا

model_choice = st.selectbox("Model", ["Baseline (fast)", "Transformer (BERT)"])

# load baseline if exists
baseline = None
if os.path.exists(baseline_path):
    try:
        baseline = joblib.load(baseline_path)
    except Exception as e:
        st.warning("Could not load baseline model: " + str(e))

# load transformer pipeline if selected
transformer_pipeline = None
if model_choice.startswith("Transformer"):
    if os.path.isdir(bert_dir):
        # load fine-tuned model from local
        try:
            tokenizer = AutoTokenizer.from_pretrained(bert_dir)
            model = AutoModelForSequenceClassification.from_pretrained(bert_dir)
            transformer_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        except Exception as e:
            st.warning("Failed to load local BERT model: " + str(e))
    # fallback to HF hub model
    if transformer_pipeline is None:
        st.info("Using pre-trained hub model: cardiffnlp/twitter-roberta-base-sentiment")
        transformer_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

# input area
text = st.text_area("Paste tweet or review here:", height=200)
if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        if model_choice.startswith("Baseline"):
            if baseline is None:
                st.error("Baseline model not found. Use Transformer or upload model.")
            else:
                pred = baseline.predict([text])[0]
                probs = None
                try:
                    probs = baseline.predict_proba([text])[0]
                except:
                    pass
                st.success(f"Prediction: {pred}")
                if probs is not None:
                    st.write("Probabilities:", probs)
        else:
            # transformer
            try:
                out = transformer_pipeline(text[:512])  # limit length
                st.success(f"Label: {out[0]['label']}  — score: {out[0]['score']:.3f}")
            except Exception as e:
                st.error("Transformer inference failed: " + str(e))

# Batch mode: upload CSV and run inference
st.markdown("---")
st.subheader("Batch inference (CSV)")
uploaded = st.file_uploader("Upload CSV with a text column named 'text'", type=["csv"])
if uploaded:
    import pandas as pd
    df = pd.read_csv(uploaded)
    if 'text' not in df:
        st.error("The column 'text' is missing in the uploaded file.")
