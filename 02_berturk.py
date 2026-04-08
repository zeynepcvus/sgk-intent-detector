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
train_dataset = SGKDataset(
    train_df["text_clean"].tolist(),
    [label2id[i] for i in train_df["intent"].tolist()],
    tokenizer
)
val_dataset = SGKDataset(
    val_df["text_clean"].tolist(),
    [label2id[i] for i in val_df["intent"].tolist()],
    tokenizer
)
test_dataset = SGKDataset(
    test_df["text_clean"].tolist(),
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
optimizer   = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
num_epochs  = 7
total_steps = len(train_loader) * num_epochs

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

best_val_f1  = 0
train_losses = []
val_losses   = []
val_f1s      = []

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
        torch.save(model.state_dict(), "models/berturk_best.pt")
        print(f"  --> Best model saved (val f1: {val_f1:.4f})")

# --- en iyi modeli yükle ve test et ---
print("\nLoading best model for test evaluation...")
model.load_state_dict(torch.load("models/berturk_best.pt", map_location=device))

test_preds, test_labels, test_loss = evaluate(model, test_loader, device)
test_acc = accuracy_score(test_labels, test_preds)
test_f1  = f1_score(test_labels, test_preds, average="macro", zero_division=0)

print(f"\nTest Accuracy : {test_acc:.4f}")
print(f"Test Macro F1 : {test_f1:.4f}")
print("\nClassification Report:")
print(classification_report(
    test_labels,
    test_preds,
    target_names=labels,
    zero_division=0
))

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