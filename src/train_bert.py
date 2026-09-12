"""Fine-tune a Chinese BERT for sentiment classification on ChnSentiCorp.

This is the script actually used to produce models/my_finetuned_bert, run on
Google Colab (free-tier T4 GPU) rather than locally — see
notebooks/train_bert_colab.ipynb for the original executed notebook with real
training/eval output. It fine-tunes
hfl/chinese-macbert-base as a binary positive/negative classifier; the neutral
class used elsewhere in this project is added afterwards via a confidence
threshold (see predict_sentiment in src/sentiment_models.py).

To reproduce on Colab:
  1. Runtime -> Change runtime type -> GPU (T4 is enough)
  2. Upload/paste this file's contents into a notebook, or `!pip install` the
     deps below and run it as a script
  3. After training, download the ./my_finetuned_bert folder (or copy it to
     Google Drive, as this script does) and place it at models/my_finetuned_bert
     in this repo

Install (Colab): !pip install transformers datasets -q
"""

import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "hfl/chinese-macbert-base"
OUTPUT_DIR = "./my_finetuned_bert"


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }


def main():
    dataset = load_dataset("lansinuote/ChnSentiCorp")
    train_texts, train_labels = list(dataset["train"]["text"]), list(dataset["train"]["label"])
    val_texts, val_labels = list(dataset["validation"]["text"]), list(dataset["validation"]["label"])
    print(f"訓練資料: {len(train_texts)} 筆, 驗證資料: {len(val_texts)} 筆")

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)

    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=200,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"模型已存到 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
