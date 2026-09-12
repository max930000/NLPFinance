"""Three sentiment classification approaches for Chinese financial news headlines:
dictionary lookup, an off-the-shelf financial BERT, and our own fine-tuned BERT.
"""

import torch
from torch.nn.functional import softmax
from transformers import BertTokenizer, BertForSequenceClassification

# ===== 1. Dictionary-based method =====
POSITIVE_WORDS = ["創高", "成長", "上揚", "獲利", "看好", "強勁"]
NEGATIVE_WORDS = ["下滑", "衰退", "虧損", "看淡", "重挫", "下跌"]


def dict_sentiment(title: str) -> int:
    """Return 1 (positive), -1 (negative) or 0 (neutral) by counting keyword hits."""
    pos = sum(word in title for word in POSITIVE_WORDS)
    neg = sum(word in title for word in NEGATIVE_WORDS)
    if pos > neg:
        return 1
    elif neg > pos:
        return -1
    return 0


# ===== 2. Off-the-shelf financial BERT =====
PRETRAINED_MODEL_NAME = "sanshizhang/Chinese-Sentiment-Analysis-Fund-Direction"
PRETRAINED_LABEL_MAP = {0: -1, 1: 1, 2: 0}  # model's 0/1/2 -> our -1/1/0 scale


def load_pretrained_bert(model_name: str = PRETRAINED_MODEL_NAME):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def bert_sentiment(text: str, tokenizer, model, label_map: dict = PRETRAINED_LABEL_MAP):
    inputs = tokenizer(text, max_length=512, padding="max_length",
                        truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probs = softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred].item()
    return label_map[pred], confidence


# ===== 3. Our fine-tuned BERT (hfl/chinese-macbert-base fine-tuned on ChnSentiCorp,
# see src/train_bert.py) — binary positive/negative, thresholded to add a neutral class =====
def load_finetuned_bert(model_path: str = "models/my_finetuned_bert"):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model


def predict_sentiment(text: str, tokenizer, model, threshold: float = 0.6) -> int:
    """Predict -1/0/1. Low-confidence predictions (< threshold) are treated as neutral,
    since the underlying model was trained as binary positive/negative only."""
    inputs = tokenizer(str(text), truncation=True, padding="max_length",
                        max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probs = softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred].item()
    if confidence < threshold:
        return 0
    return 1 if pred == 1 else -1
