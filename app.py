import streamlit as st

from src.sentiment_models import (
    dict_sentiment,
    load_pretrained_bert,
    bert_sentiment,
    load_finetuned_bert,
    predict_sentiment,
)

st.set_page_config(page_title="財經新聞情緒分析", layout="wide")


@st.cache_resource
def _load_pretrained():
    return load_pretrained_bert()


@st.cache_resource
def _load_finetuned():
    return load_finetuned_bert("models/my_finetuned_bert")


label_display = {1: "🟢 正面", -1: "🔴 負面", 0: "⚪ 中立"}

st.title("📈 財經新聞情緒分析比較系統")
st.caption("比較詞典法 / 現成財經 BERT / 自己 fine-tune 的 BERT 對新聞標題的情緒判斷")

text_input = st.text_area(
    "輸入一則財經新聞標題",
    "台積電營收創新高，市場看好後市表現",
)

if st.button("分析", type="primary"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("詞典法")
        result = dict_sentiment(text_input)
        st.metric("情緒判斷", label_display[result])
        st.caption("測試集準確率: 27%")

    with col2:
        st.subheader("現成財經 BERT")
        tokenizer, model = _load_pretrained()
        result, confidence = bert_sentiment(text_input, tokenizer, model)
        st.metric("情緒判斷", label_display[result])
        st.caption(f"信心分數: {confidence:.2%} ／ 測試集準確率: 55%")

    with col3:
        st.subheader("Fine-tuned BERT")
        tokenizer, model = _load_finetuned()
        result = predict_sentiment(text_input, tokenizer, model)
        st.metric("情緒判斷", label_display[result])
        st.caption("測試集準確率: 67%")
