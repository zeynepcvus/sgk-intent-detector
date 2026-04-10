import pandas as pd
import re
import os

# --- veri yükleme ---
# İyileştirilmiş v4 dataset: 873 satır, 11 intent (out_of_scope dahil)
df = pd.read_excel("data/sgk_dataset_improved_v4.xlsx")
print(f"Raw data loaded: {len(df)} rows")
print(f"Intents ({df['intent'].nunique()}): {sorted(df['intent'].unique())}")

# --- temizleme fonksiyonu (baseline TF-IDF için) ---
# NOT: BERTurk cased model için bu temizleme KULLANILMAZ.
# BERTurk orijinal text'i (text_bert sütunu) alır — büyük/küçük harf
# ve noktalama bilgisi korunur.
def clean_text(text):
    # küçük harfe çevir
    text = text.lower()
    # noktalama işaretlerini kaldır
    text = re.sub(r"[^\w\s]", " ", text)
    # birden fazla boşluğu tek boşluğa indir
    text = re.sub(r"\s+", " ", text)
    # baş ve sondaki boşlukları sil
    text = text.strip()
    return text

# --- baseline için agresif temizleme ---
df["text_clean"] = df["text"].apply(clean_text)

# --- BERT için hafif temizleme (sadece baş/son boşluk) ---
# BERTurk cased modeli büyük/küçük harf ve noktalamaya duyarlı.
# Orijinal metni korumak daha iyi temsil sağlar.
df["text_bert"] = df["text"].str.strip()

# --- önce / sonra karşılaştırma ---
print("\nSample cleaning results:")
print("-" * 60)
for _, row in df.sample(5, random_state=42).iterrows():
    print(f"Original  : {row['text']}")
    print(f"BERT input: {row['text_bert']}")
    print(f"Baseline  : {row['text_clean']}")
    print()

print("Split distribution:")
print(df["split"].value_counts().to_string())
print("\nIntent distribution:")
print(df["intent"].value_counts().to_string())

# --- temizlenmiş veriyi kaydet ---
os.makedirs("data", exist_ok=True)
df.to_csv("data/sgk_dataset_clean.csv", index=False, encoding="utf-8-sig")
print(f"\nCleaned data saved: data/sgk_dataset_clean.csv")
print(f"Total rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")