import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_excel('data/sgk_dataset_final.xlsx')

# ============================================================
# STEP 1: FIX MISLABELED EXAMPLES
# ============================================================

fixes = {
    "sgk'da aktif görünüyorum ama sağlık hizmetine erişemiyorum neden": "health_coverage_query",
    "sigorta primleri düzenli yatırılmış mı": "premium_days_query",
    "primlerim eksiksiz yatırılmış mı": "premium_days_query",
}

print("=== STEP 1: LABEL FIXES ===")
fixed_count = 0
for text, new_intent in fixes.items():
    mask = df['text'] == text
    if mask.any():
        old_intent = df.loc[mask, 'intent'].values[0]
        df.loc[mask, 'intent'] = new_intent
        print(f"FIXED: {old_intent} → {new_intent}")
        print(f"  Text: {text}")
        fixed_count += 1
    else:
        print(f"NOT FOUND: {text}")

print(f"\nTotal fixes: {fixed_count}")

# ============================================================
# STEP 2: NEW TARGETED EXAMPLES
# Format: (text, intent, style, length_bucket)
# ============================================================

new_examples = [

    # --- insurance_status_query (10 new) ---
    # ANCHOR: sistemde kayıt/tescil durumu (NO sağlık/hastane/provizyon keywords)
    ("adıma resmi sigorta tescili yapılıp yapılmadığını kontrol etmek istiyorum", "insurance_status_query", "normal", "uzun"),
    ("işverenimin beni sgk sistemine bildirip bildirmediğini öğrenmek istiyorum", "insurance_status_query", "normal", "uzun"),
    ("sisteme işlendim mi henüz", "insurance_status_query", "günlük", "kısa"),
    ("isim soyismim sgkda gözüküyor mu", "insurance_status_query", "günlük", "kısa"),
    ("4a kapsamında sigortalı olup olmadığımı sorgulamak istiyorum", "insurance_status_query", "normal", "orta"),
    ("yeni işe başladım girişim yapılmış mı sgkda", "insurance_status_query", "bozuk", "orta"),
    ("üstüme herhangi bir sigorta tescili var mı sadece kayıt durumunu merak ediyorum", "insurance_status_query", "normal", "uzun"),
    ("sgkda gözüküyom mu bilmiyorum girişim oldu mu", "insurance_status_query", "bozuk", "orta"),
    ("işverenimi beni sgkya bildirdi mi tescilim var mı", "insurance_status_query", "günlük", "orta"),
    ("sigorta numarama kayıtlı aktif tescil bulunuyor mu öğrenmek istiyorum", "insurance_status_query", "normal", "orta"),

    # --- health_coverage_query (10 new) ---
    # ANCHOR: hastane/muayene/provizyon/müstahaklık/ilaç
    ("eczaneden ilaç alacağım sgk provizyonum açık mı kontrol etmek istiyorum", "health_coverage_query", "normal", "uzun"),
    ("sgk müstahaklık sorgusunu nasıl yapabilirim", "health_coverage_query", "normal", "orta"),
    ("hastaneye gitsem kapıda provizyon alınır mı sgk'dan", "health_coverage_query", "günlük", "orta"),
    ("provizyon var mı üstümde", "health_coverage_query", "günlük", "kısa"),
    ("acil servise gitsem sgk masrafı karşılar mi", "health_coverage_query", "bozuk", "orta"),
    ("müstahaklığım var mı sgk sisteminde", "health_coverage_query", "günlük", "orta"),
    ("provizyonum acik mi doktora gitmeden önce kontrol etmek istiyorum", "health_coverage_query", "bozuk", "orta"),
    ("sevk almadan özel hastaneye gidebilir miyim sgk karşılar mı", "health_coverage_query", "normal", "orta"),
    ("sgk kaydım var biliyorum ama hastanede sağlık hakkım geçerli mi onu sormak istiyorum", "health_coverage_query", "normal", "uzun"),
    ("yeni işe girdim sağlık hizmetlerinden ne zaman yararlanabileceğimi öğrenmek istiyorum", "health_coverage_query", "normal", "uzun"),

    # --- service_record_query (10 new) ---
    # ANCHOR: hangi işyerleri, çalışma tarihleri, döküm belgesi - NOT kaç gün
    ("geçmişte hangi işverenlere bağlı çalıştığımı liste halinde görmek istiyorum", "service_record_query", "normal", "orta"),
    ("sgk'da kayıtlı işyeri ve çalışma dönemlerimi gösteren belgeye ihtiyacım var", "service_record_query", "normal", "uzun"),
    ("daha önce nerede çalıştığıma sgk'dan bakabilir miyim", "service_record_query", "günlük", "orta"),
    ("sgk'daki iş geçmişimi görmek istiyorum kaç gün değil hangi işyerleri", "service_record_query", "günlük", "uzun"),
    ("gectigim isyerlerini sgkdan listelemek istiyorum", "service_record_query", "bozuk", "orta"),
    ("hizmet cetveli almak istiyorum sgkdan", "service_record_query", "bozuk", "kısa"),
    ("hangi şirkette ne zaman çalıştığımı gösteren dökümü almak istiyorum", "service_record_query", "günlük", "orta"),
    ("sigortam olan tüm işverenler kimler bunu görmek istiyorum", "service_record_query", "normal", "orta"),
    ("gün sayısıyla değil sadece işyeri listesiyle ilgileniyorum sgk hizmet dökümü", "service_record_query", "normal", "uzun"),
    ("4a hizmet cetvelimi resmi olarak almak istiyorum", "service_record_query", "normal", "orta"),

    # --- premium_days_query (10 new) ---
    # ANCHOR: toplam gün sayısı/rakam, prim günleri sayısı - NOT hangi işyeri
    ("toplam prim gün sayım emekliliğe yetecek mi öğrenmek istiyorum", "premium_days_query", "normal", "orta"),
    ("sgk'da birikmiş toplam gün sayım kaç oldu", "premium_days_query", "normal", "orta"),
    ("kaç günüm var şu an sgk'da", "premium_days_query", "günlük", "kısa"),
    ("gün sayım yeterli mi emeklilik için", "premium_days_query", "günlük", "kısa"),
    ("toplam kac gun primim birikmis bakabilirmiyim", "premium_days_query", "bozuk", "orta"),
    ("prim gün sayımı öğrenmek istiyorum işyerleriyle değil sadece rakam", "premium_days_query", "normal", "orta"),
    ("kac gunluk sigortam var sgkda", "premium_days_query", "bozuk", "kısa"),
    ("4a ve 4b kapsamındaki toplam sigortalılık gün sayım ne kadar", "premium_days_query", "normal", "orta"),
    ("birikmiş günlerim kaç tane sgk'da", "premium_days_query", "günlük", "kısa"),
    ("farklı dönemlerden toplam kaç prim günüm oluştuğunu sorgulayabilir miyim", "premium_days_query", "normal", "uzun"),

    # --- sgk_debt_query (10 new) ---
    # ANCHOR: borç var mı, borç miktarı sorgusu - NOT ödeme yapmak
    ("adıma kayıtlı herhangi bir sgk borcu var mı sorgulamak istiyorum", "sgk_debt_query", "normal", "orta"),
    ("e-devlet'ten sgk borç durumumu nasıl öğrenebilirim", "sgk_debt_query", "normal", "orta"),
    ("borç birikmiş mi sgk'da öğrenmek istiyorum", "sgk_debt_query", "günlük", "kısa"),
    ("ödeme yapmayacağım sadece borcum olup olmadığına bakacağım", "sgk_debt_query", "günlük", "orta"),
    ("borc cikmis mi üzerime sgk sisteminde", "sgk_debt_query", "bozuk", "kısa"),
    ("gecikmiş prim borcum bulunup bulunmadığını sorgulamak istiyorum", "sgk_debt_query", "normal", "orta"),
    ("prim borcum var mı kontrol etmek istiyorum henüz ödeme değil sadece sorgulama", "sgk_debt_query", "normal", "uzun"),
    ("sgk borc sorgulama nasıl yapılır", "sgk_debt_query", "bozuk", "orta"),
    ("borcum var mı görmek istiyorum ödemeden önce kontrol edeyim", "sgk_debt_query", "günlük", "orta"),
    ("birikmiş sgk borcumu rakam olarak görmek istiyorum ne kadar çıkmış", "sgk_debt_query", "normal", "orta"),

    # --- premium_payment_query (10 new) ---
    # ANCHOR: ödeme yapmak, prim yatırmak, nasıl ödenir - NOT var mı sorgusu
    ("sgk borcumu hemen ödemek istiyorum en hızlı yol nedir", "premium_payment_query", "normal", "orta"),
    ("borcumu biliyorum artık ödeme adımına geçmek istiyorum", "premium_payment_query", "normal", "orta"),
    ("ödeme ekranına geçmek istiyorum sgk için", "premium_payment_query", "günlük", "kısa"),
    ("zaten borcum var onu ödemek istiyorum nereden yapayım", "premium_payment_query", "günlük", "orta"),
    ("sgk borcumu odeyecegim nasıl yapıcam", "premium_payment_query", "bozuk", "orta"),
    ("sgk prim borcumu kredi kartıyla ödeyebilir miyim", "premium_payment_query", "normal", "orta"),
    ("otomatik ödeme talimatı vererek sgk primimin düzenli kesilmesini sağlamak istiyorum", "premium_payment_query", "normal", "uzun"),
    ("borc odeme sayfasina gitmek istiyorum sgk", "premium_payment_query", "bozuk", "orta"),
    ("sgk borcumu nasıl ödeyebilirim adım adım göster", "premium_payment_query", "günlük", "orta"),
    ("e-devlet üzerinden sgk borç ödeme işlemini başlatmak istiyorum", "premium_payment_query", "normal", "uzun"),

    # --- out_of_scope (25 new) ---
    # SGK ile ilgisi olmayan konular — eDevlet dışı
    ("ehliyet almak için hangi belgeler gerekiyor", "out_of_scope", "normal", "orta"),
    ("vergi borcumu nasıl öğrenebilirim", "out_of_scope", "normal", "kısa"),
    ("sürücü belgemi yenilemek istiyorum nereye gideyim", "out_of_scope", "günlük", "orta"),
    ("okul kaydı için nereye başvurmalıyım", "out_of_scope", "normal", "kısa"),
    ("belediyeye su faturası ödeyecektim nasıl yapıyorum", "out_of_scope", "günlük", "orta"),
    ("tapu senedi çıkarmak istiyorum hangi ofise gideyim", "out_of_scope", "normal", "orta"),
    ("ehliyete girecegim ne lazim", "out_of_scope", "bozuk", "kısa"),
    ("icra takibi başlatmak istiyorum nasıl yapabilirim", "out_of_scope", "normal", "orta"),
    ("pasaport randevusu almak istiyorum nereye başvurayım", "out_of_scope", "günlük", "orta"),
    ("kira sözleşmesi için noter onayı gerekiyor mu", "out_of_scope", "normal", "orta"),
    ("vergi levhası nereden alınır", "out_of_scope", "normal", "kısa"),
    ("trafik cezasını ödemek istiyorum nasıl yapıyorum", "out_of_scope", "bozuk", "orta"),
    ("araç muayenesi ne zaman yaptırmalıyım", "out_of_scope", "günlük", "kısa"),
    ("mahkeme duruşma tarihini nasıl öğrenebilirim", "out_of_scope", "normal", "orta"),
    ("elektrik faturasına itiraz nasıl yapılır", "out_of_scope", "günlük", "orta"),
    ("okul kaydimi yenilemek istiyorum nereden", "out_of_scope", "bozuk", "kısa"),
    ("işletme ruhsatı almak için ne yapmalıyım", "out_of_scope", "normal", "orta"),
    ("ikamet belgesi nereden alınır nüfus müdürlüğünden mi", "out_of_scope", "günlük", "orta"),
    ("haciz kaldırmak için hangi adımları izlemeliyim", "out_of_scope", "normal", "orta"),
    ("ehliyet yenileme icin hangi evraklar gerekli", "out_of_scope", "bozuk", "orta"),
    ("gelir vergisi beyannamesi nasıl verilir hangi tarihler", "out_of_scope", "normal", "orta"),
    ("yargı sisteminde dava takibi nasıl yapılır", "out_of_scope", "bozuk", "orta"),
    ("çocuğumun okul kaydı için evrak listesi lazım", "out_of_scope", "günlük", "orta"),
    ("nüfus cüzdanı yenilemek için nereye gideyim", "out_of_scope", "günlük", "kısa"),
    ("belediye imar izni nasıl alınır", "out_of_scope", "normal", "kısa"),
]

# ============================================================
# STEP 3: BUILD NEW DATAFRAME & VERIFY
# ============================================================

new_df = pd.DataFrame(new_examples, columns=['text', 'intent', 'style', 'length_bucket'])
new_df['split'] = 'train'

print("\n=== STEP 2: NEW EXAMPLES SUMMARY ===")
print(f"New examples to add: {len(new_df)}")
print("\nNew examples per intent:")
print(new_df['intent'].value_counts().to_string())

# Apply label fixes to original df
for text, new_intent in fixes.items():
    df.loc[df['text'] == text, 'intent'] = new_intent

# Check for duplicates with existing data
existing_texts = set(df['text'].str.strip().str.lower())
new_texts_lower = new_df['text'].str.strip().str.lower()
dups = new_df[new_texts_lower.isin(existing_texts)]
if len(dups) > 0:
    print(f"\nDUPLICATE WARNING - {len(dups)} conflicts:")
    print(dups[['text', 'intent']].to_string())
else:
    print("\nNo duplicates with existing data.")

# ============================================================
# STEP 4: COMBINE & SAVE
# ============================================================

df_improved = pd.concat([df, new_df], ignore_index=True)

print("\n=== STEP 3: FINAL DATASET STATS ===")
print(f"Original size: {len(df)}")
print(f"New examples: {len(new_df)}")
print(f"Final size: {len(df_improved)}")
print("\nIntent distribution:")
print(df_improved['intent'].value_counts().to_string())
print("\nStyle distribution:")
print(df_improved['style'].value_counts().to_string())
print("\nSplit distribution:")
print(df_improved['split'].value_counts().to_string())

# Save
output_path = 'data/sgk_dataset_improved.xlsx'
df_improved.to_excel(output_path, index=False)
print(f"\nSaved to: {output_path}")
print("Original file UNTOUCHED: data/sgk_dataset_final.xlsx")
