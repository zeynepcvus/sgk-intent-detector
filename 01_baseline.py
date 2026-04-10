import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# --- veri yükleme ---
df = pd.read_csv("data/sgk_dataset_clean.csv", encoding="utf-8-sig")

# train, validation ve test setlerini ayır
train_df = df[df["split"] == "train"]
val_df   = df[df["split"] == "validation"]
test_df  = df[df["split"] == "test"]

X_train = train_df["text_clean"].tolist()
y_train = train_df["intent"].tolist()

X_val   = val_df["text_clean"].tolist()
y_val   = val_df["intent"].tolist()

X_test  = test_df["text_clean"].tolist()
y_test  = test_df["intent"].tolist()

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# --- TF-IDF vektörleştirici ---
# her kelimeye ve 2'li kelime çiftine göre özellik çıkarır (tf-idf her cümleyi sayısal bir vektöre çevirir)
vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),   # unigram + bigram
    max_features=5000,    # en sık 5000 özellik
    sublinear_tf=True     # aşırı tekrar eden kelimeleri bastır
)

X_train_tfidf = vectorizer.fit_transform(X_train)   # öğren + dönüştür (hangi kelimeler önemli öğrenir sayıya çevirir)
X_val_tfidf   = vectorizer.transform(X_val)         # sadece dönüştür (model ondan öğrenmemeli)
X_test_tfidf  = vectorizer.transform(X_test)        # sadece dönüştür

# --- Logistic Regression modeli ---
# Her cümlenin sayısal hali var elimizde. Bu sayılar hangi intent sorusunu cevaplıyoruz.

model = LogisticRegression(
    max_iter=1000,  # öğrenmek için 1000 adım at
    C=1.0,          # regularization (ne kadar katı öğrensin)
    solver="lbfgs", # hesaplama yöntemi
)

model.fit(X_train_tfidf, y_train)

# --- validation seti değerlendirme ---
val_preds = model.predict(X_val_tfidf)
val_acc   = accuracy_score(y_val, val_preds)
print(f"\nValidation Accuracy: {val_acc:.4f}")

# --- test seti değerlendirme ---
test_preds = model.predict(X_test_tfidf)
test_acc   = accuracy_score(y_test, test_preds)
print(f"Test Accuracy: {test_acc:.4f}")

test_f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)
print(f"Test Macro F1: {test_f1:.4f}")

print("\nClassification Report:")
report_str = classification_report(y_test, test_preds, zero_division=0)
print(report_str)

# --- evaluation sonuçlarını JSON olarak kaydet ---
report_dict = classification_report(y_test, test_preds, output_dict=True, zero_division=0)
labels_sorted = sorted(df["intent"].unique())

from collections import defaultdict
confusion_pairs = defaultdict(int)
for true, pred in zip(y_test, test_preds):
    if true != pred:
        confusion_pairs[f"{true} → {pred}"] += 1
top_confusions = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)[:10]

eval_summary = {
    "model": "baseline",
    "model_name": "TF-IDF + Logistic Regression",
    "num_intents": len(labels_sorted),
    "test_accuracy": round(test_acc, 4),
    "test_macro_f1": round(test_f1, 4),
    "per_class_f1": {
        label: round(report_dict[label]["f1-score"], 4)
        for label in labels_sorted if label in report_dict
    },
    "top_confusions": top_confusions,
}

os.makedirs("outputs", exist_ok=True)
summary_path = "outputs/evaluation_summary.json"
combined = {}
if os.path.exists(summary_path):
    with open(summary_path) as f:
        combined = json.load(f)
combined["baseline"] = eval_summary

with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(combined, f, ensure_ascii=False, indent=2)
print(f"Evaluation summary saved: {summary_path}")

# --- confusion matrix kaydet ---
os.makedirs("outputs", exist_ok=True)

labels = sorted(df["intent"].unique())
cm = confusion_matrix(y_test, test_preds, labels=labels)

plt.figure(figsize=(12, 9))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=labels,
    yticklabels=labels,
    cmap="Blues"
)
plt.title("Baseline (TF-IDF + LR) — Confusion Matrix")
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("outputs/baseline_confusion_matrix.png", dpi=150)
plt.close()
print("\nConfusion matrix has been saved: outputs/baseline_confusion_matrix.png")

# --- modeli kaydet ---
os.makedirs("models", exist_ok=True)
with open("models/baseline_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
print("Model has been saved: models/baseline_model.pkl")