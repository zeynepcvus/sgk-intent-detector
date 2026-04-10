import pandas as pd
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_excel('data/sgk_dataset_improved_v3.xlsx')
original_len = len(df)
print(f"Loaded: {original_len} rows\n")

log = []

# ============================================================
# PROBLEM 1: out_of_scope split dağılımını düzelt
# 25 örnek, tamamı train → hedef: test'te en az 5
# 25 * 80/10/10 = 20/2.5/2.5 (5 karşılanmaz)
# 25 * 60/20/20 = 15/5/5 → test=5, val=5 ✓
# ============================================================

oos_idx = df[df['intent'] == 'out_of_scope'].index.tolist()
print(f"out_of_scope toplam: {len(oos_idx)}")

# Rastgele ama tekrar üretilebilir
rng = np.random.default_rng(seed=42)
shuffled = rng.permutation(oos_idx).tolist()

n_total    = len(shuffled)   # 25
n_val      = 5
n_test     = 5
n_train    = n_total - n_val - n_test  # 15

train_idx  = shuffled[:n_train]
val_idx    = shuffled[n_train:n_train + n_val]
test_idx   = shuffled[n_train + n_val:]

df.loc[train_idx, 'split'] = 'train'
df.loc[val_idx,   'split'] = 'validation'
df.loc[test_idx,  'split'] = 'test'

# Doğrula
oos_after = df[df['intent'] == 'out_of_scope']['split'].value_counts()
log.append(f"PROBLEM 1 — out_of_scope split güncellendi:")
log.append(f"  train:      {oos_after.get('train',0)}")
log.append(f"  validation: {oos_after.get('validation',0)}")
log.append(f"  test:       {oos_after.get('test',0)}")

# ============================================================
# PROBLEM 2: premium_days içindeki emeklilik cümleleri
# ============================================================

# 3 tespit edilen örnek ve kararlar:
#
# 1) "toplam prim gün sayım emekliliğe yetecek mi öğrenmek istiyorum"
#    → "emekliliğe yetecek mi" = emeklilik şartı sorgusu → retirement_query
#
# 2) "gün sayım yeterli mi emeklilik için"
#    → "emeklilik için yeterli mi" = emeklilik şartı sorgusu → retirement_query
#
# 3) "emeklilik için toplamda kaç günüm eksik onu öğrenmek istiyorum"
#    → "emeklilik için kaç günüm eksik" = emeklilik koşul sorgusu → retirement_query

relabels = [
    ("toplam prim gün sayım emekliliğe yetecek mi öğrenmek istiyorum",   "retirement_query"),
    ("gün sayım yeterli mi emeklilik için",                               "retirement_query"),
    ("emeklilik için toplamda kaç günüm eksik onu öğrenmek istiyorum",    "retirement_query"),
]

log.append("\nPROBLEM 2 — premium_days → retirement_query:")
for text, new_intent in relabels:
    mask = df['text'] == text
    if mask.any():
        df.loc[mask, 'intent'] = new_intent
        log.append(f"  LABEL FIXED → {new_intent}: {text}")
    else:
        log.append(f"  NOT FOUND: {text}")

# ============================================================
# ÇIKTI
# ============================================================

print("=== DEĞİŞİKLİK LOGU ===")
for entry in log:
    print(entry)

print("\n=== FINAL STATS ===")
print(f"Toplam satır: {original_len} → {len(df)}  (değişmedi: {len(df)==original_len})")

print("\nIntent dağılımı:")
print(df['intent'].value_counts().to_string())

print("\nGenel split dağılımı:")
print(df['split'].value_counts().to_string())

print("\nout_of_scope split:")
print(df[df['intent']=='out_of_scope']['split'].value_counts().to_string())

print(f"\nExact duplicates: {df.duplicated(subset=['text']).sum()}")

df_orig = pd.read_excel('data/sgk_dataset_final.xlsx')
print(f"Orijinal dosya korundu: {len(df_orig)==780}")

# Save
out = 'data/sgk_dataset_improved_v4.xlsx'
df.to_excel(out, index=False)
print(f"\nKaydedildi: {out}")
