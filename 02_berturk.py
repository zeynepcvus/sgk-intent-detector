import json
import pandas as pd
import torch
import random
import numpy as np
import os
import pickle
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- tekrar üretilebilirlik için seed ---
# aynı seed ile her çalıştırmada aynı sonucu alırız
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# --- cihaz ayarı ---
# gpu varsa gpu, yoksa cpu kullan
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# --- veri yükleme ---
df = pd.read_csv("data/sgk_dataset_clean.csv", encoding="utf-8-sig")

train_df = df[df["split"] == "train"]
val_df   = df[df["split"] == "validation"]
test_df  = df[df["split"] == "test"]

# --- etiket dönüşümü ---
# model sayılarla çalışır, intent isimlerini sayıya çeviriyoruz
labels   = sorted(df["intent"].unique())
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

num_labels = len(labels)
print(f"Number of intents: {num_labels}")
print(f"Labels: {labels}")

# --- tokenizer ve model yükle ---
# BertTokenizer yerine AutoTokenizer kullanmak daha esnek
MODEL_NAME = "dbmdz/bert-base-turkish-cased"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)

# --- dataset sınıfı ---
# pytorch'a veriyi nasıl vereceğimizi tanımlıyoruz
# max_len=128: intent cümleleri kısa olduğu için 128 token yeterli
class SGKDataset(Dataset):
    def __init__(self, texts, intents, tokenizer, max_len=128):
        self.texts     = texts
        self.intents   = intents
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # her cümleyi tokenize edip sayılara çevir
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label":          torch.tensor(self.intents[idx], dtype=torch.long)
        }

# --- dataset ve dataloader oluştur ---
# BERTurk cased model: orijinal text kullan (text_bert), text_clean DEĞİL.
# Lowercase + punctuation removal cased BERT için bilgi kaybına yol açar.
# text_bert = sadece strip() uygulanmış, büyük/küçük harf ve noktalama korunmuş.
BERT_TEXT_COL = "text_bert" if "text_bert" in train_df.columns else "text_clean"
print(f"BERT input column: '{BERT_TEXT_COL}'")

train_dataset = SGKDataset(
    train_df[BERT_TEXT_COL].tolist(),
    [label2id[i] for i in train_df["intent"].tolist()],
    tokenizer
)
val_dataset = SGKDataset(
    val_df[BERT_TEXT_COL].tolist(),
    [label2id[i] for i in val_df["intent"].tolist()],
    tokenizer
)
test_dataset = SGKDataset(
    test_df[BERT_TEXT_COL].tolist(),
    [label2id[i] for i in test_df["intent"].tolist()],
    tokenizer
)

# dataloader: veriyi batch batch modele verir
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=16, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=16, shuffle=False)

# --- model yükle ---
# BertForSequenceClassification yerine AutoModel kullanmak daha esnek
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels
)
model.to(device)

# --- optimizer ve scheduler ---
optimizer       = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
num_epochs      = 10        # maksimum epoch — early stopping devreye girebilir
EARLY_STOP_PAT  = 3         # val F1 iyileşmezse kaç epoch bekle
total_steps     = len(train_loader) * num_epochs

# öğrenme hızını yavaş yavaş düşüren scheduler
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=total_steps // 10,
    num_training_steps=total_steps
)

# --- eğitim fonksiyonu ---
def train_epoch(model, loader, optimizer, scheduler, device):
    # modeli eğitim moduna al
    model.train()
    total_loss = 0

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_batch   = batch["label"].to(device)

        # gradyanları sıfırla
        optimizer.zero_grad()

        # forward pass: modelden tahmin al
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels_batch
        )

        loss = outputs.loss

        # backward pass: gradyanları hesapla
        loss.backward()

        # gradyan patlamasını önle
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # ağırlıkları güncelle
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# --- değerlendirme fonksiyonu ---
def evaluate(model, loader, device):
    # modeli değerlendirme moduna al
    model.eval()
    all_preds  = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch   = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels_batch
            )

            # validation loss da izle
            total_loss += outputs.loss.item()

            # en yüksek skoru alan sınıfı seç
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())

    avg_loss = total_loss / len(loader)
    return all_preds, all_labels, avg_loss


# --- eğitim döngüsü ---
print("\nTraining started...")
print("-" * 50)

best_val_f1    = 0
no_improve     = 0      # early stopping sayacı
train_losses   = []
val_losses     = []
val_f1s        = []
best_epoch     = 0

for epoch in range(num_epochs):
    # eğit
    train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
    train_losses.append(train_loss)

    # validation değerlendirmesi
    val_preds, val_labels, val_loss = evaluate(model, val_loader, device)
    val_losses.append(val_loss)

    # accuracy yerine macro f1 ile best model seç
    # macro f1: tüm sınıflara eşit ağırlık verir, daha adil
    val_f1  = f1_score(val_labels, val_preds, average="macro", zero_division=0)
    val_acc = accuracy_score(val_labels, val_preds)
    val_f1s.append(val_f1)

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

    # en iyi f1 veren modeli kaydet
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_epoch  = epoch + 1
        no_improve  = 0
        torch.save(model.state_dict(), "models/berturk_best.pt")
        print(f"  --> Best model saved (val f1: {val_f1:.4f})")
    else:
        no_improve += 1
        if no_improve >= EARLY_STOP_PAT:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {EARLY_STOP_PAT} epochs)")
            break

print(f"\nBest epoch: {best_epoch} | Best val F1: {best_val_f1:.4f}")

# --- en iyi modeli yükle ve test et ---
print("\nLoading best model for test evaluation...")
model.load_state_dict(torch.load("models/berturk_best.pt", map_location=device))

test_preds, test_labels, test_loss = evaluate(model, test_loader, device)
test_acc = accuracy_score(test_labels, test_preds)
test_f1  = f1_score(test_labels, test_preds, average="macro", zero_division=0)

print(f"\nTest Accuracy : {test_acc:.4f}")
print(f"Test Macro F1 : {test_f1:.4f}")
print("\nClassification Report:")
report_str = classification_report(
    test_labels,
    test_preds,
    target_names=labels,
    zero_division=0
)
print(report_str)

# --- evaluation sonuçlarını JSON olarak kaydet ---
report_dict = classification_report(
    test_labels,
    test_preds,
    target_names=labels,
    output_dict=True,
    zero_division=0
)

# karışan sınıf çiftlerini tespit et (hata analizi için)
from collections import defaultdict
confusion_pairs = defaultdict(int)
for true, pred in zip(
    [labels[i] for i in test_labels],
    [labels[i] for i in test_preds]
):
    if true != pred:
        confusion_pairs[f"{true} → {pred}"] += 1
top_confusions = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)[:10]

eval_summary = {
    "model": "berturk",
    "model_name": "dbmdz/bert-base-turkish-cased",
    "bert_input": BERT_TEXT_COL,
    "best_epoch": best_epoch,
    "num_intents": num_labels,
    "test_accuracy": round(test_acc, 4),
    "test_macro_f1": round(test_f1, 4),
    "per_class_f1": {
        label: round(report_dict[label]["f1-score"], 4)
        for label in labels
    },
    "top_confusions": top_confusions,
}

os.makedirs("outputs", exist_ok=True)

# Eğer baseline summary varsa onu da dahil et
summary_path = "outputs/evaluation_summary.json"
combined = {}
if os.path.exists(summary_path):
    with open(summary_path) as f:
        combined = json.load(f)
combined["berturk"] = eval_summary

with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(combined, f, ensure_ascii=False, indent=2)
print(f"\nEvaluation summary saved: {summary_path}")

# --- confusion matrix ---
os.makedirs("outputs", exist_ok=True)
cm = confusion_matrix(test_labels, test_preds)

plt.figure(figsize=(12, 9))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=labels,
    yticklabels=labels,
    cmap="Purples"
)
plt.title("BERTurk Fine-tuned — Confusion Matrix")
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("outputs/berturk_confusion_matrix.png", dpi=150)
plt.close()
print("Confusion matrix saved: outputs/berturk_confusion_matrix.png")

# --- etiket eşleştirmesini kaydet ---
# gradio arayüzünde kullanmak için
with open("models/label2id.pkl", "wb") as f:
    pickle.dump(label2id, f)
with open("models/id2label.pkl", "wb") as f:
    pickle.dump(id2label, f)

print("Label mappings saved.")
print("\nDone!")