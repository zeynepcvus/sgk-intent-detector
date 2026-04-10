"""
evaluate_models.py
------------------
Eğitilmiş baseline ve BERTurk modellerini test seti üzerinde değerlendirir.
Yeni eğitim yapmaz — mevcut model dosyalarını kullanır.

Çıktı:
  outputs/evaluation_summary.json  — makine tarafından okunabilir metrikler
  outputs/baseline_confusion_matrix.png
  outputs/berturk_confusion_matrix.png
  (terminale) karşılaştırmalı tablo

Kullanım:
  python scripts/evaluate_models.py
"""

import json
import os
import pickle
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# 1. Veri yükle
# ---------------------------------------------------------------------------
DATA_PATH = "data/sgk_dataset_clean.csv"
if not os.path.exists(DATA_PATH):
    print(f"ERROR: {DATA_PATH} bulunamadı. Önce 00_preprocess.py çalıştırın.")
    sys.exit(1)

df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
test_df = df[df["split"] == "test"].reset_index(drop=True)
print(f"Test seti: {len(test_df)} örnek")
print(f"Intent dağılımı:\n{test_df['intent'].value_counts().to_string()}\n")

labels_sorted = sorted(df["intent"].unique())
y_test = test_df["intent"].tolist()

# BERT için text sütunu: önce text_bert, yoksa text, son çare text_clean
BERT_COL = "text_bert" if "text_bert" in test_df.columns else (
    "text" if "text" in test_df.columns else "text_clean"
)
print(f"BERT input column: '{BERT_COL}'")

os.makedirs("outputs", exist_ok=True)

# ---------------------------------------------------------------------------
# 2. Baseline değerlendirme
# ---------------------------------------------------------------------------
def evaluate_baseline():
    if not os.path.exists("models/baseline_model.pkl"):
        print("Baseline model bulunamadı, atlanıyor.")
        return None, None

    with open("models/baseline_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    X_test = vectorizer.transform(test_df["text_clean"].tolist())
    preds  = model.predict(X_test)
    acc    = accuracy_score(y_test, preds)
    f1     = f1_score(y_test, preds, average="macro", zero_division=0)

    print("=" * 55)
    print("BASELINE — TF-IDF + Logistic Regression")
    print("=" * 55)
    print(f"Test Accuracy : {acc:.4f}")
    print(f"Test Macro F1 : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_test, preds, labels=labels_sorted)
    plt.figure(figsize=(12, 9))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels_sorted,
                yticklabels=labels_sorted, cmap="Blues")
    plt.title("Baseline (TF-IDF + LR) — Confusion Matrix")
    plt.ylabel("True label"); plt.xlabel("Predicted label")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    plt.savefig("outputs/baseline_confusion_matrix.png", dpi=150)
    plt.close()
    print("Confusion matrix saved: outputs/baseline_confusion_matrix.png")

    return acc, f1, preds

# ---------------------------------------------------------------------------
# 3. BERTurk değerlendirme
# ---------------------------------------------------------------------------
def evaluate_berturk():
    if not os.path.exists("models/berturk_best.pt"):
        print("BERTurk model bulunamadı, atlanıyor.")
        return None, None

    with open("models/id2label.pkl", "rb") as f:
        id2label = pickle.load(f)
    with open("models/label2id.pkl", "rb") as f:
        label2id = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nBERTurk device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "dbmdz/bert-base-turkish-cased", num_labels=len(id2label)
    )
    model.load_state_dict(torch.load("models/berturk_best.pt", map_location=device))
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    texts  = test_df[BERT_COL].tolist()
    intents = test_df["intent"].tolist()

    # Batch inference (batch_size=32)
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_lbls  = intents[i:i+batch_size]
            enc = tokenizer(
                batch_texts, max_length=128, padding="max_length",
                truncation=True, return_tensors="pt"
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            preds_batch = torch.argmax(out.logits, dim=1).cpu().numpy()
            all_preds.extend([id2label[p] for p in preds_batch])
            all_labels.extend(batch_lbls)

    # Sadece model'in bildiği label'ları filtrele
    known_labels = set(id2label.values())
    valid_idx    = [i for i, l in enumerate(all_labels) if l in known_labels]
    y_true_f = [all_labels[i] for i in valid_idx]
    y_pred_f = [all_preds[i]  for i in valid_idx]
    skipped  = len(all_labels) - len(valid_idx)
    if skipped:
        print(f"  NOTE: {skipped} örnek modelin bilmediği bir intent'e ait, atlandı.")

    acc  = accuracy_score(y_true_f, y_pred_f)
    f1   = f1_score(y_true_f, y_pred_f, average="macro", zero_division=0)

    known_labels_sorted = sorted(known_labels)
    print("\n" + "=" * 55)
    print("BERTurk — Fine-tuned Turkish BERT")
    print("=" * 55)
    print(f"Test Accuracy : {acc:.4f}")
    print(f"Test Macro F1 : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true_f, y_pred_f,
                                labels=known_labels_sorted, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_true_f, y_pred_f, labels=known_labels_sorted)
    plt.figure(figsize=(12, 9))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=known_labels_sorted,
                yticklabels=known_labels_sorted, cmap="Purples")
    plt.title("BERTurk Fine-tuned — Confusion Matrix")
    plt.ylabel("True label"); plt.xlabel("Predicted label")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    plt.savefig("outputs/berturk_confusion_matrix.png", dpi=150)
    plt.close()
    print("Confusion matrix saved: outputs/berturk_confusion_matrix.png")

    return acc, f1, y_true_f, y_pred_f

# ---------------------------------------------------------------------------
# 4. Karşılaştırmalı özet
# ---------------------------------------------------------------------------
def build_summary(b_acc, b_f1, b_preds, bert_acc, bert_f1, bert_true, bert_preds):
    summary = {}

    if b_acc is not None:
        conf_pairs = defaultdict(int)
        for t, p in zip(y_test, b_preds):
            if t != p:
                conf_pairs[f"{t} → {p}"] += 1
        top = sorted(conf_pairs.items(), key=lambda x: x[1], reverse=True)[:10]

        report_d = classification_report(y_test, b_preds, output_dict=True, zero_division=0)
        summary["baseline"] = {
            "model": "baseline",
            "model_name": "TF-IDF + Logistic Regression",
            "num_intents": len(labels_sorted),
            "test_accuracy": round(b_acc, 4),
            "test_macro_f1": round(b_f1, 4),
            "per_class_f1": {
                lb: round(report_d[lb]["f1-score"], 4)
                for lb in labels_sorted if lb in report_d
            },
            "top_confusions": top,
        }

    if bert_acc is not None:
        known_labels = sorted(set(bert_true))
        conf_pairs = defaultdict(int)
        for t, p in zip(bert_true, bert_preds):
            if t != p:
                conf_pairs[f"{t} → {p}"] += 1
        top = sorted(conf_pairs.items(), key=lambda x: x[1], reverse=True)[:10]

        report_d = classification_report(bert_true, bert_preds,
                                         labels=known_labels,
                                         output_dict=True, zero_division=0)
        summary["berturk"] = {
            "model": "berturk",
            "model_name": "dbmdz/bert-base-turkish-cased (fine-tuned)",
            "bert_input": BERT_COL,
            "num_intents": len(known_labels),
            "test_accuracy": round(bert_acc, 4),
            "test_macro_f1": round(bert_f1, 4),
            "per_class_f1": {
                lb: round(report_d[lb]["f1-score"], 4)
                for lb in known_labels if lb in report_d
            },
            "top_confusions": top,
        }

    path = "outputs/evaluation_summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nEvaluation summary saved: {path}")

    # Karşılaştırma tablosu
    if b_acc is not None and bert_acc is not None:
        print("\n" + "=" * 40)
        print("MODEL KARŞILAŞTIRMASI")
        print("=" * 40)
        print(f"{'Model':<28} {'Accuracy':>9} {'Macro F1':>9}")
        print("-" * 40)
        print(f"{'TF-IDF + LR (Baseline)':<28} {b_acc:>9.4f} {b_f1:>9.4f}")
        print(f"{'BERTurk Fine-tuned':<28} {bert_acc:>9.4f} {bert_f1:>9.4f}")
        delta_acc = bert_acc - b_acc
        delta_f1  = bert_f1 - b_f1
        print("-" * 40)
        print(f"{'Delta (BERT - Baseline)':<28} {delta_acc:>+9.4f} {delta_f1:>+9.4f}")

    return summary


# ---------------------------------------------------------------------------
# 5. Çalıştır
# ---------------------------------------------------------------------------
b_result    = evaluate_baseline()
bert_result = evaluate_berturk()

b_acc  = b_result[0]  if b_result else None
b_f1   = b_result[1]  if b_result else None
b_pred = b_result[2]  if b_result else None

bert_acc  = bert_result[0] if bert_result else None
bert_f1   = bert_result[1] if bert_result else None
bert_true = bert_result[2] if bert_result else None
bert_pred = bert_result[3] if bert_result else None

build_summary(b_acc, b_f1, b_pred, bert_acc, bert_f1, bert_true, bert_pred)
