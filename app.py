import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

# إعداد الصفحة
st.set_page_config(page_title="🧠 Tweet Sentiment Analyzer", layout="wide")
st.title("🧠 Tweet Sentiment Analyzer")
st.write("حلل مشاعر التغريدات باستخدام الذكاء الاصطناعي")

# تحميل النموذج
@st.cache_resource
def load_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

sentiment_analyzer = load_model()

# إدخال النص
tweet_text = st.text_area("✍️ اكتب التغريدة هنا", height=100)

if st.button("تحليل المشاعر"):
    if tweet_text.strip():
        result = sentiment_analyzer(tweet_text)
        label = result[0]['label']
        score = round(result[0]['score'], 3)

        st.success(f"📌 النتيجة: **{label}** (الثقة: {score})")
    else:
        st.warning("⚠️ الرجاء إدخال نص قبل التحليل")

# رفع ملف CSV
uploaded_file = st.file_uploader("📂 أو ارفع ملف CSV يحتوي على عمود 'text'", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if 'text' in df.columns:
        st.write("🔍 تحليل جميع التغريدات في الملف...")
        df['sentiment'] = df['text'].apply(lambda x: sentiment_analyzer(str(x))[0]['label'])
        st.dataframe(df)
        csv_output = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 تحميل النتائج CSV", data=csv_output, file_name="tweet_sentiment_results.csv", mime="text/csv")
    else:
        st.error("❌ الملف يجب أن يحتوي على عمود اسمه 'text'")
