import pandas as pd
import re
import os

# --- veri yükleme ---
df = pd.read_excel("data/sgk_dataset_final.xlsx")
print(f"Raw data loaded: {len(df)} rows")

# --- temizleme fonksiyonu ---
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

# --- temizlemeyi uygula ---
df["text_clean"] = df["text"].apply(clean_text)

# --- önce / sonra karşılaştırma ---
print("\nSample cleaning results:")
print("-" * 60)
for _, row in df.sample(5, random_state=42).iterrows():
    print(f"Original : {row['text']}")
    print(f"Cleaned  : {row['text_clean']}")
    print()

# --- temizlenmiş veriyi kaydet ---
os.makedirs("data", exist_ok=True)
df.to_csv("data/sgk_dataset_clean.csv", index=False, encoding="utf-8-sig")
print(f"Cleaned data saved: data/sgk_dataset_clean.csv")
print(f"Total rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")