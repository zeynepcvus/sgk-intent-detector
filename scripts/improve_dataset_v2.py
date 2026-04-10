import pandas as pd
import sys
from difflib import SequenceMatcher

sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_excel('data/sgk_dataset_improved.xlsx')
original_len = len(df)
print(f"Loaded: {original_len} rows\n")

changes_log = []

# ============================================================
# CHANGE 1: LABEL FIX
# "sgk prim günlerinin dökümünü görmek istiyorum"
#  → premium_days_query YANLIŞ, service_record_query OLMALI
#  Neden: "döküm görmek" = service_record anchor; odak sayı değil liste/belge
# ============================================================
mask = df['text'] == 'sgk prim günlerinin dökümünü görmek istiyorum'
if mask.any():
    df.loc[mask, 'intent'] = 'service_record_query'
    changes_log.append("LABEL FIX: 'sgk prim günlerinin dökümünü görmek istiyorum'  premium_days → service_record  (döküm=liste odağı, sayı değil)")
    print(f"[1] LABEL FIX applied")
else:
    print("[1] LABEL FIX — text not found")

# ============================================================
# CHANGE 2: REMOVE NEAR-DUPLICATE (service_record)
# "nerelerde çalıştım sgk a göre" [günlük]
#  SIM=0.95 ile "nerelerde çalıştım sgk'ya göre" [günlük]
#  Sadece apostrof farkı — bozuk varyant "nerelerde çalışmışım sgk ya gore" zaten var
# ============================================================
mask2 = (df['text'] == "nerelerde çalıştım sgk a göre")
if mask2.any():
    df = df[~mask2].reset_index(drop=True)
    changes_log.append("REMOVE (near-dup SIM=0.95): 'nerelerde çalıştım sgk a göre'  —  'nerelerde çalıştım sgk'ya göre' [günlük] zaten mevcut, apostrof farkı anlamsız")
    print(f"[2] REMOVE near-dup applied — service_record")
else:
    print("[2] REMOVE — text not found")

# ============================================================
# CHANGE 3: REMOVE NEAR-DUPLICATE (premium_days)
# "sgk da kac gunum var" [bozuk]
#  SIM=0.97 ile "sgkda kac gunum var" [bozuk]
#  Sadece boşluk farkı — anlam özdeş
# ============================================================
mask3 = (df['text'] == "sgk da kac gunum var")
if mask3.any():
    df = df[~mask3].reset_index(drop=True)
    changes_log.append("REMOVE (near-dup SIM=0.97): 'sgk da kac gunum var'  —  'sgkda kac gunum var' [bozuk] zaten mevcut, sadece boşluk farkı")
    print(f"[3] REMOVE near-dup applied — premium_days")
else:
    print("[3] REMOVE — text not found")

# ============================================================
# CHANGE 4: REMOVE NEAR-DUPLICATE (service_record — v1 eklentisi)
# "4a hizmet cetvelimi resmi olarak almak istiyorum" [normal]
#  SIM=0.95 ile "sgk hizmet cetvelimi resmi olarak almak istiyorum" [normal]
#  v1'de eklendi ama orijinalle neredeyse aynı
# ============================================================
mask4 = (df['text'] == "4a hizmet cetvelimi resmi olarak almak istiyorum")
if mask4.any():
    df = df[~mask4].reset_index(drop=True)
    changes_log.append("REMOVE (near-dup SIM=0.95, v1 eklentisi): '4a hizmet cetvelimi resmi olarak almak istiyorum'  —  orijinal 'sgk hizmet cetvelimi resmi olarak almak istiyorum' ile özdeş")
    print(f"[4] REMOVE near-dup applied — service_record (v1 addition)")
else:
    print("[4] REMOVE — text not found")

print(f"\nAfter removals/fixes: {len(df)} rows")

# ============================================================
# CHANGE 5: KÜÇÜK HEDEFLI EKLEMELER (2-3 per critical pair)
# Amaç: sınır cümlelerini güçlendirmek
# ============================================================

targeted_new = [

    # --- service_record vs premium_days ayrımı ---
    # service_record: tarih/işyeri odağı, sayı değil
    ("her işyerindeki çalışma dönemini başlangıç bitiş tarihiyle görmek istiyorum",
     "service_record_query", "normal", "uzun"),
    ("hangi firmalarda hangi tarihlerde çalıştım bunu öğrenmek istiyorum kaç gün değil",
     "service_record_query", "günlük", "uzun"),

    # premium_days: rakam/toplam odağı, işyeri değil
    ("emeklilik için toplamda kaç günüm eksik onu öğrenmek istiyorum",
     "premium_days_query", "normal", "orta"),
    ("1080 günü geçtim mi henüz toplam prim günümü kontrol etmek istiyorum",
     "premium_days_query", "normal", "orta"),

    # --- registration_document vs service_record ayrımı ---
    # registration_document: resmi belge/çıktı odağı, geçmiş değil
    ("e-devletten sgk kaydımı gösteren barkodlu resmi belgeyi indirmek istiyorum",
     "registration_document_query", "normal", "orta"),
    ("sgk sigortalılık tescil belgesinin barkodlu çıktısını almak istiyorum",
     "registration_document_query", "normal", "orta"),

    # --- insurance_status vs health_coverage ayrımı ---
    # insurance_status: kayıt/tescil sorgusu, sağlık değil
    ("sgk'da sigortam var mı sadece kayıt durumumu merak ediyorum hastane ile ilgili değil",
     "insurance_status_query", "normal", "uzun"),

    # health_coverage: hastanede kullanım, kayıt değil
    ("kaydım zaten var ama hastanede geçerli mi onu sormak istiyorum",
     "health_coverage_query", "günlük", "orta"),

    # --- sgk_debt vs premium_payment ayrımı ---
    # sgk_debt: sorgulama, ödeme eylemi yok
    ("bu ay primim düşmemiş borcum oluşmuş mu diye kontrol etmek istiyorum",
     "sgk_debt_query", "günlük", "orta"),

    # premium_payment: ödeme eylemi, sorgu değil
    ("borcumu biliyorum hangi ödeme kanalını kullansam daha hızlı olur",
     "premium_payment_query", "normal", "orta"),
]

new_df = pd.DataFrame(targeted_new, columns=['text', 'intent', 'style', 'length_bucket'])
new_df['split'] = 'train'

# Duplicate check
existing_texts = set(df['text'].str.strip().str.lower())
new_lower = new_df['text'].str.strip().str.lower()
dups = new_df[new_lower.isin(existing_texts)]
if len(dups) > 0:
    print(f"\nDUP WARNING — {len(dups)} already exist:")
    print(dups[['text', 'intent']].to_string())
else:
    print(f"\nNew targeted examples: {len(new_df)} — no duplicates")

changes_log.append(f"ADD: {len(new_df)} targeted boundary-clarifying examples (2-3 per critical pair)")
for _, row in new_df.iterrows():
    changes_log.append(f"  + [{row['intent']}] {row['text']}")

df_v2 = pd.concat([df, new_df], ignore_index=True)

# ============================================================
# FINAL STATS
# ============================================================
print("\n=== CHANGE LOG ===")
for c in changes_log:
    print(c)

print("\n=== FINAL STATS ===")
print(f"v1 size:  {original_len}")
print(f"v2 size:  {len(df_v2)}")
print(f"Net diff: {len(df_v2) - original_len:+d}")
print("\nIntent distribution:")
print(df_v2['intent'].value_counts().to_string())
print("\nStyle distribution:")
print(df_v2['style'].value_counts().to_string())
print("\nSplit distribution:")
print(df_v2['split'].value_counts().to_string())

# Verify originals untouched
df_orig = pd.read_excel('data/sgk_dataset_final.xlsx')
print(f"\nOriginal file rows: {len(df_orig)} (UNCHANGED: {len(df_orig)==780})")

# ============================================================
# SAVE
# ============================================================
output_path = 'data/sgk_dataset_improved_v2.xlsx'
df_v2.to_excel(output_path, index=False)
print(f"\nSaved: {output_path}")
print("Previous versions untouched:")
print("  data/sgk_dataset_final.xlsx    (780 rows, orijinal)")
print("  data/sgk_dataset_improved.xlsx (865 rows, v1)")
