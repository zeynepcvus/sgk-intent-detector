import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_excel('data/sgk_dataset_improved_v2.xlsx')
original_len = len(df)
print(f"Loaded: {original_len} rows\n")

log = []

# ============================================================
# PHASE 1+2: YAPAY ÖRNEKLER — SİL veya YENİDEN YAZ
# ============================================================

# --- SİLİNECEKLER (çok yapay, gerçek kullanıcı böyle yazmaz) ---

deletes = [
    # "kaç gün değil" meta-açıklaması — v2'de ekledim, fazla yapay
    "sgk'daki iş geçmişimi görmek istiyorum kaç gün değil hangi işyerleri",
    # "kaç gün değil" tekrar — v2'de ekledim
    "hangi firmalarda hangi tarihlerde çalıştım bunu öğrenmek istiyorum kaç gün değil",
    # "hastane ile ilgili değil" meta-açıklaması — v2'de ekledim, yapay
    "sgk'da sigortam var mı sadece kayıt durumumu merak ediyorum hastane ile ilgili değil",
    # "ödeme yapmayacağım" + "sadece sorgulama" — v2'de ekledim, çok yapay
    "ödeme yapmayacağım sadece borcum olup olmadığına bakacağım",
    # "henüz ödeme değil sadece sorgulama" — v2'de ekledim, çok yapay
    "prim borcum var mı kontrol etmek istiyorum henüz ödeme değil sadece sorgulama",
]

for text in deletes:
    mask = df['text'] == text
    if mask.any():
        intent = df.loc[mask, 'intent'].values[0]
        df = df[~mask].reset_index(drop=True)
        log.append(f"SİLİNDİ [{intent}]: {text}")
    else:
        log.append(f"NOT FOUND (delete): {text}")

# --- YENİDEN YAZILACAKLAR (intent doğru, ifade yapay) ---
# Eski metni güncelle — in-place rewrite

rewrites = [
    # "sadece kayıt durumunu merak ediyorum" açıklayıcı kısım gereksiz
    (
        "üstüme herhangi bir sigorta tescili var mı sadece kayıt durumunu merak ediyorum",
        "üstüme herhangi bir sigorta tescili var mı",
        "insurance_status_query"
    ),
    # "kaydım zaten var ama ... onu sormak istiyorum" kalıbı yapay
    (
        "kaydım zaten var ama hastanede geçerli mi onu sormak istiyorum",
        "hastane randevusu aldım sgk kapsamım aktif mi",
        "health_coverage_query"
    ),
    # "zaten borcum var onu" açıklayıcı başlangıç
    (
        "zaten borcum var onu ödemek istiyorum nereden yapayım",
        "borcumu ödemek istiyorum nasıl yapabilirim",
        "premium_payment_query"
    ),
    # "borcumu biliyorum hangi ödeme kanalını" — meta çerçeveleme
    (
        "borcumu biliyorum hangi ödeme kanalını kullansam daha hızlı olur",
        "sgk borç ödemesi için en pratik yol nedir",
        "premium_payment_query"
    ),
]

for old_text, new_text, expected_intent in rewrites:
    mask = df['text'] == old_text
    if mask.any():
        actual_intent = df.loc[mask, 'intent'].values[0]
        # Verify intent matches before rewriting
        if actual_intent == expected_intent:
            df.loc[mask, 'text'] = new_text
            log.append(f"YENİDEN YAZILDI [{expected_intent}]:")
            log.append(f"  ESKİ: {old_text}")
            log.append(f"  YENİ: {new_text}")
        else:
            log.append(f"INTENT MISMATCH (skip rewrite): expected {expected_intent}, found {actual_intent}: {old_text}")
    else:
        log.append(f"NOT FOUND (rewrite): {old_text}")

print(f"After deletes+rewrites: {len(df)} rows")

# ============================================================
# PHASE 3: SERVICE_RECORD vs REGISTRATION_DOCUMENT güçlendirme
# + silinen yapay debt örnekleri için doğal yedekler
# ============================================================

# Etiketi değiştirilecek örnek var mı?
# "kamu kurumu başvurusu için sgk sigortalılık belgesi ve hizmet dökümü lazım"
# → registration_document'ta — "hizmet dökümü" kelimesi var ama asıl niyet
#    "kuruma belge sunmak" → registration_document baskın, bırakıyorum.
# "sgk belgesi mi hizmet dökümü mü istendi tam emin değilim ikisini de alayım"
# → registration_document — belirsiz kullanıcı niyet. Bırakıyorum.
# Hiçbir etiket değişikliği gerekmedi bu turda.
log.append("\nETİKET DEĞİŞİKLİĞİ: Gerekli bulunmadı — service_record/reg_doc sınırı mevcut durumda kabul edilebilir")

# Küçük hedefli eklemeler
targeted_adds = [
    # sgk_debt için silinen 2 yapay örneğin doğal yedeği
    ("üzerime prim borcu var mı öğrenmek istiyorum",
     "sgk_debt_query", "normal", "kısa"),
    ("gss prim borcum birikmiş mi kontrol etmek istiyorum",
     "sgk_debt_query", "günlük", "orta"),

    # service_record için silinen 2 yapay örneğin doğal yedeği
    ("hangi tarihlerde hangi işyerlerinde sigortalı çalıştığımı listelemek istiyorum",
     "service_record_query", "normal", "orta"),
    ("sgk kayıtlarımda geçmiş işverenlerimi ve çalışma dönemlerimi görmek istiyorum",
     "service_record_query", "normal", "uzun"),

    # insurance_status için silinen yapay örneğin doğal yedeği
    ("sgk sisteminde aktif sigorta kaydım görünüyor mu",
     "insurance_status_query", "normal", "kısa"),

    # registration_document — phase 3 güçlendirme (1 yeni, net anchor)
    ("resmi kuruma sunmak için sigortalılık durumumu gösteren belge almak istiyorum",
     "registration_document_query", "normal", "uzun"),
]

new_df = pd.DataFrame(targeted_adds, columns=['text', 'intent', 'style', 'length_bucket'])
new_df['split'] = 'train'

# Duplicate check
existing_lower = set(df['text'].str.strip().str.lower())
new_lower = new_df['text'].str.strip().str.lower()
dups = new_df[new_lower.isin(existing_lower)]
if len(dups) > 0:
    print(f"DUP WARNING {len(dups)}:")
    print(dups[['text','intent']].to_string())
else:
    print(f"New additions: {len(new_df)} — no duplicates")

for _, row in new_df.iterrows():
    log.append(f"EKLENDİ [{row['intent']}] [{row['style']}]: {row['text']}")

df_v3 = pd.concat([df, new_df], ignore_index=True)

# ============================================================
# ÇIKTI
# ============================================================
print("\n=== DEĞİŞİKLİK LOGU ===")
for entry in log:
    print(entry)

print("\n=== FINAL STATS ===")
print(f"v2: {original_len} → v3: {len(df_v3)}  (net {len(df_v3)-original_len:+d})")
print("\nIntent dağılımı:")
print(df_v3['intent'].value_counts().to_string())
print("\nStyle dağılımı:")
print(df_v3['style'].value_counts().to_string())

# Sanity checks
print(f"\nExact duplicates: {df_v3.duplicated(subset=['text']).sum()}")
print(f"Çok kısa (<12 kar): {(df_v3['text'].str.len() < 12).sum()}")

df_orig = pd.read_excel('data/sgk_dataset_final.xlsx')
print(f"Orijinal dosya korundu: {len(df_orig)==780}")

# Save
out = 'data/sgk_dataset_improved_v3.xlsx'
df_v3.to_excel(out, index=False)
print(f"\nKaydedildi: {out}")
print("Korunanlar:")
print("  data/sgk_dataset_final.xlsx        (780, orijinal)")
print("  data/sgk_dataset_improved.xlsx     (865, v1)")
print("  data/sgk_dataset_improved_v2.xlsx  (872, v2)")
