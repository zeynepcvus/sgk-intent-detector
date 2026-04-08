import re
import pickle
from typing import Dict, List, Tuple

import numpy as np
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(
    page_title="SGK Akıllı Niyet Tespit Sistemi",
    page_icon="🛡️",
    layout="wide",
)

INTENT_META: Dict[str, Dict[str, str]] = {
    "insurance_status_query": {
        "title": "Sigorta Durumu Sorgulama",
        "description": "Kullanıcı, aktif sigorta kaydının sistemde görünüp görünmediğini öğrenmek istiyor.",
        "action": "Sigorta tescil ve hizmet bilgilerinizi e-Devlet üzerinden kontrol edebilirsiniz.",
        "link_label": "Sigorta Durumunu Kontrol Et",
        "link": "https://www.turkiye.gov.tr/sgk-tescil-ve-hizmet-dokumu",
        "icon": "🪪",
    },
    "premium_days_query": {
        "title": "Prim Gün Sayısı Sorgulama",
        "description": "Kullanıcı, toplam prim gün sayısını veya biriken gün bilgisini görmek istiyor.",
        "action": "Prim gün sayınızı hizmet dökümü ekranı üzerinden inceleyebilirsiniz.",
        "link_label": "Prim Günlerini Gör",
        "link": "https://www.turkiye.gov.tr/4a-hizmet-dokumu",
        "icon": "📊",
    },
    "sgk_debt_query": {
        "title": "SGK Borç Sorgulama",
        "description": "Kullanıcı, SGK veya GSS borcu olup olmadığını öğrenmek istiyor.",
        "action": "Borç durumunuzu ilgili GSS/SGK borç sorgulama ekranından kontrol edebilirsiniz.",
        "link_label": "Borç Durumunu Gör",
        "link": "https://www.turkiye.gov.tr/sgk-gss-borc-dokumu",
        "icon": "💳",
    },
    "retirement_query": {
        "title": "Emeklilik Uygunluk Sorgulama",
        "description": "Kullanıcı, emeklilik şartlarını, tarihini veya uygunluk durumunu öğrenmek istiyor.",
        "action": "Normal şartlarda ne zaman emekli olabileceğinizi bu ekran üzerinden öğrenebilirsiniz.",
        "link_label": "Emeklilik Bilgilerini Gör",
        "link": "https://www.turkiye.gov.tr/sgk-ne-zaman-emekli-olabilirim",
        "icon": "⏳",
    },
    "health_coverage_query": {
        "title": "Sağlık Kapsamı Sorgulama",
        "description": "Kullanıcı, sağlık güvencesinin aktif olup olmadığını ve hastanede geçerli olup olmadığını öğrenmek istiyor.",
        "action": "Sağlık provizyonu ve müstehaklık durumunuzu bu ekran üzerinden kontrol edebilirsiniz.",
        "link_label": "Sağlık Kapsamını Kontrol Et",
        "link": "https://www.turkiye.gov.tr/spas-mustahaklik-sorgulama",
        "icon": "🏥",
    },
    "employment_status_query": {
        "title": "İşe Giriş / Çıkış Durumu",
        "description": "Kullanıcı, işe giriş veya çıkış bilgisinin sisteme işlenip işlenmediğini kontrol etmek istiyor.",
        "action": "İşe giriş ve işten ayrılış kayıtlarınızı bu ekran üzerinden görüntüleyebilirsiniz.",
        "link_label": "Çalışma Kaydını Gör",
        "link": "https://www.turkiye.gov.tr/sosyal-guvenlik-4a-ise-giris-cikis-bildirgesi",
        "icon": "💼",
    },
    "service_record_query": {
        "title": "Hizmet Dökümü / Çalışma Geçmişi",
        "description": "Kullanıcı, çalışma geçmişini ve hizmet dökümünü görüntülemek istiyor.",
        "action": "Hizmet dökümünüzü e-Devlet üzerinden görüntüleyebilir ve çıktı alabilirsiniz.",
        "link_label": "Hizmet Dökümünü Aç",
        "link": "https://www.turkiye.gov.tr/4a-hizmet-dokumu",
        "icon": "📄",
    },
    "registration_document_query": {
        "title": "Kayıt / Sigortalılık Belgesi",
        "description": "Kullanıcı, resmi belge almak veya sigortalılık durumunu belgelemek istiyor.",
        "action": "Barkodlu sosyal güvenlik kayıt belgenizi bu ekran üzerinden alabilirsiniz.",
        "link_label": "Belgeyi Görüntüle",
        "link": "https://www.turkiye.gov.tr/sosyal-guvenlik-kayit-belgesi-sorgulama",
        "icon": "🧾",
    },
    "premium_payment_query": {
        "title": "Prim / Borç Ödeme İşlemi",
        "description": "Kullanıcı, borcunu ödemek veya ödeme ekranına ulaşmak istiyor.",
        "action": "Prim veya borç ödeme işlemini kart ile ödeme ekranı üzerinden başlatabilirsiniz.",
        "link_label": "Ödeme Ekranına Git",
        "link": "https://www.turkiye.gov.tr/sosyal-guvenlik-sosyal-guvenlik-kurumu-kart-ile-prim-odeme-uygulamasi",
        "icon": "💰",
    },
    "general_info_query": {
        "title": "Genel Bilgi Talebi",
        "description": "Kullanıcı, SGK sistemi veya işlemler hakkında genel bilgi almak istiyor.",
        "action": "Tüm SGK ve sigorta hizmetlerini bu liste üzerinden inceleyebilirsiniz.",
        "link_label": "SGK Hizmetlerini Gör",
        "link": "https://www.turkiye.gov.tr/sosyal-guvenlik-ve-sigorta-hizmetleri",
        "icon": "ℹ️",
    },
}

EXAMPLE_QUERIES: List[Tuple[str, str]] = [
    ("Sigortam aktif mi?", "Sigortam aktif mi öğrenmek istiyorum"),
    ("Prim gün sayım kaç?", "Prim gün sayım kaç olmuş"),
    ("SGK borcum var mı?", "SGK borcum var mı"),
    ("Emeklilik yaşım kaç?", "Emeklilik yaşım kaç?"),
    ("Hizmet dökümümü görmek istiyorum", "Hizmet dökümümü görmek istiyorum"),
    ("Sağlık güvencem geçerli mi?", "Sağlık güvencem hastanede geçerli mi"),
]
EXAMPLE_CHIP_ROW: List[Tuple[str, str]] = EXAMPLE_QUERIES[:4]

MAX_INPUT_CHARS = 256


def _set_example_query(query: str) -> None:
    st.session_state.user_input_area = query


def confidence_donut_svg(pct: int) -> str:
    r = 38
    circ = 2 * 3.14159265 * r
    dash = max(0.0, min(1.0, pct / 100.0)) * circ
    return f"""<svg width="128" height="128" viewBox="0 0 100 100" class="conf-donut-svg" aria-hidden="true">
  <circle cx="50" cy="50" r="{r}" fill="none" stroke="#dcfce7" stroke-width="8"/>
  <circle cx="50" cy="50" r="{r}" fill="none" stroke="#16a34a" stroke-width="8"
    stroke-dasharray="{dash:.2f} {circ:.2f}" stroke-linecap="round" transform="rotate(-90 50 50)"/>
  <text x="50" y="56" text-anchor="middle" font-size="18" font-weight="800" fill="#15803d" font-family="system-ui,sans-serif">%{pct}</text>
</svg>"""


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_resource
def load_models():
    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
    with open("models/id2label.pkl", "rb") as f:
        id2label = pickle.load(f)
    with open("models/label2id.pkl", "rb") as f:
        label2id = pickle.load(f)
    bert_model = AutoModelForSequenceClassification.from_pretrained(
        "dbmdz/bert-base-turkish-cased",
        num_labels=len(id2label)
    )
    bert_model.load_state_dict(
        torch.load("models/berturk_best.pt", map_location=torch.device("cpu"))
    )
    bert_model.eval()
    with open("models/baseline_model.pkl", "rb") as f:
        baseline_model = pickle.load(f)
    with open("models/tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return tokenizer, bert_model, id2label, baseline_model, vectorizer


def predict_bert(text, tokenizer, model, id2label):
    cleaned = clean_text(text)
    inputs = tokenizer(cleaned, return_tensors="pt", max_length=128, padding="max_length", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1).squeeze().numpy()
    top_indices = np.argsort(probs)[::-1]
    all_results = [(id2label[i], float(probs[i])) for i in top_indices]
    return all_results[0][0], all_results[0][1], all_results[:5]


def predict_baseline(text, model, vectorizer):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    return model.predict(vec)[0]


def conf_label(conf):
    if conf >= 0.85: return "Çok Yüksek Güven"
    elif conf >= 0.70: return "Yüksek Güven"
    elif conf >= 0.50: return "Orta Güven"
    else: return "Düşük Güven"


# --- YENİ: domain kontrolü ve prediction status ---
SGK_KEYWORDS = [
    "sgk", "sigorta", "prim", "emekli", "borç", "hizmet dökümü",
    "işe giriş", "işten çıkış", "sağlık güvence", "e-devlet",
    "sigortalı", "gün sayı", "tescil", "bağkur", "gss",
    "işveren", "bildiri", "provizyon", "aylık", "tahsilat",
    "borcum", "sigortam", "primim", "günüm", "kaydım",
    "hastane", "muayene", "döküm", "belge", "kayıt"
]


def is_sgk_related(text: str) -> bool:
    # metinde sgk anahtar kelimelerinden en az biri geçiyor mu
    text_lower = text.lower()
    return any(kw in text_lower for kw in SGK_KEYWORDS)


def get_prediction_status(text: str, confidence: float, top_results: list) -> str:
    # sgk ile alakasız → out_of_domain
    if not is_sgk_related(text):
        return "out_of_domain"
    # confidence çok düşükse → belirsiz
    if confidence < 0.55:
        return "uncertain"
    # top1 ve top2 farkı çok azsa → belirsiz
    if len(top_results) >= 2:
        top2_conf = top_results[1][1]
        if confidence - top2_conf < 0.15:
            return "uncertain"
    return "normal"


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, .stApp { font-family: 'Inter', system-ui, sans-serif; }
body { background: #e8eef9; }
.stApp {
    background:
        radial-gradient(ellipse 90% 55% at 50% -15%, rgba(79, 70, 229, 0.14), transparent 55%),
        radial-gradient(ellipse 70% 40% at 100% 20%, rgba(37, 99, 235, 0.10), transparent 50%),
        radial-gradient(ellipse 60% 35% at 0% 30%, rgba(99, 102, 241, 0.08), transparent 45%),
        linear-gradient(180deg, #dce4f7 0%, #e8edfa 28%, #eef2ff 52%, #f4f6fb 100%);
}
.navbar {
    background: #0f2d6b;
    padding: 18px 30px;
    border-radius: 0 0 17px 17px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: -1rem -4rem 0.85rem -4rem;
    box-shadow: 0 4px 20px rgba(15, 45, 107, 0.25);
}
@media (max-width: 900px) {
    .navbar { margin-left: -1rem; margin-right: -1rem; border-radius: 0 0 12px 12px; padding: 15px 17px; flex-wrap: wrap; gap: 11px; }
}
.nav-left { display: flex; align-items: center; gap: 15px; }
.nav-logo { background: white; color: #0f2d6b; font-weight: 800; font-size: 15px; padding: 9px 15px; border-radius: 11px; letter-spacing: 0.02em; }
.nav-title { color: white; font-size: 18px; font-weight: 700; letter-spacing: -0.02em; }
.nav-right { display: flex; gap: 11px; flex-wrap: wrap; align-items: center; }
.nav-pill-gray { background: rgba(255,255,255,0.2); color: #e2e8f0; font-size: 13px; font-weight: 600; padding: 7px 15px; border-radius: 999px; border: 1px solid rgba(255,255,255,0.28); }
.nav-pill-blue { background: #2563eb; color: white; font-size: 13px; font-weight: 600; padding: 7px 15px; border-radius: 999px; border: 1px solid rgba(255,255,255,0.35); box-shadow: 0 2px 8px rgba(37, 99, 235, 0.35); }
.hero { text-align: center; padding: 3px 14px 8px; max-width: 800px; margin: 0 auto -2px auto; }
.hero h1 { font-size: clamp(26px, 3.8vw, 36px); font-weight: 800; color: #0a2463; margin: 0 0 7px 0; letter-spacing: -0.03em; line-height: 1.2; }
.hero .hero-lead { font-size: 16px; font-weight: 600; color: #475569; line-height: 1.45; margin: 0 auto; max-width: none; white-space: nowrap; }
@media (max-width: 900px) { .hero .hero-lead { white-space: normal; max-width: 34em; line-height: 1.5; } }
.st-key-input_card { background: rgba(255,255,255,0.96) !important; border-radius: 17px !important; padding: 16px 20px 18px !important; border: 1px solid rgba(226,232,240,0.95) !important; box-shadow: 0 4px 28px rgba(15,45,107,0.07), 0 1px 4px rgba(15,23,42,0.05) !important; margin-top: 2px !important; }
.st-key-input_card [data-testid="stVerticalBlock"] { gap: 0.22rem !important; }
.st-key-examples_chips_row { margin-top: 4px !important; padding-top: 0 !important; }
.st-key-input_card [data-testid="stHorizontalBlock"] { margin-bottom: 0 !important; }
.st-key-input_card .st-key-examples_chips_row [data-testid="stVerticalBlock"] { gap: 0 !important; }
.st-key-examples_chips_row [data-testid="stHorizontalBlock"] { gap: 8px !important; align-items: center !important; flex-wrap: nowrap !important; }
.st-key-examples_chips_row [data-testid="column"] { padding-top: 0 !important; padding-bottom: 0 !important; }
.st-key-examples_chips_row [data-testid="column"]:first-child > div { display: flex !important; align-items: center !important; justify-content: flex-start !important; min-height: 36px !important; height: 100% !important; width: 100% !important; }
.st-key-examples_chips_row [data-testid="column"]:first-child [data-testid="stMarkdownContainer"] { width: 100% !important; display: flex !important; justify-content: flex-start !important; align-items: center !important; min-height: 36px !important; }
.st-key-examples_chips_row [data-testid="column"]:not(:first-child) > div { display: flex !important; align-items: center !important; min-height: 36px !important; }
.st-key-examples_chips_row [data-testid="column"]:not(:first-child) .stButton { width: 100% !important; display: flex !important; align-items: center !important; margin-top: 0 !important; margin-bottom: 0 !important; }
.st-key-examples_chips_row [data-testid="column"]:first-child [data-testid="stMarkdownContainer"] p { margin: 0 !important; }
@media (max-width: 900px) { .st-key-examples_chips_row [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; row-gap: 9px !important; } }
.examples-inline-label { display: inline-flex; align-items: center; justify-content: flex-start; gap: 7px; margin: 0; padding: 0; white-space: nowrap; line-height: 1; height: 36px; min-height: 36px; box-sizing: border-box; }
.examples-inline-label .ex-sparkle-svg { display: block; flex-shrink: 0; width: 14px; height: 14px; align-self: center; }
.examples-inline-label .ex-label-text { font-size: 14px; font-weight: 500; color: #334155; line-height: 1; }
.st-key-examples_chips_row .stButton > button:not([kind="primary"]) { width: 100% !important; background: #e9f0ff !important; color: #2563eb !important; border: none !important; border-radius: 999px !important; padding: 8px 16px !important; font-size: 14px !important; font-weight: 500 !important; height: 36px !important; min-height: 36px !important; display: flex !important; align-items: center !important; justify-content: center !important; box-shadow: none !important; white-space: nowrap !important; overflow: hidden !important; text-overflow: ellipsis !important; }
.st-key-examples_chips_row .stButton > button:not([kind="primary"]) > div { overflow: hidden !important; text-overflow: ellipsis !important; white-space: nowrap !important; width: 100% !important; text-align: center !important; }
.st-key-examples_chips_row .stButton > button:not([kind="primary"]):hover { background: #dbe7ff !important; }
.st-key-result_panel { background: #fff !important; border-radius: 17px !important; padding: 20px 22px 24px !important; border: 1px solid #e2e8f0 !important; box-shadow: 0 4px 24px rgba(15,23,42,0.06) !important; margin-top: 12px !important; }
.result-head { display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 11px; margin-bottom: 17px; }
.result-head-left { display: flex; align-items: center; gap: 9px; font-size: 17px; font-weight: 700; color: #1e293b; }
.badge-instant { background: #dcfce7; color: #15803d; font-size: 12px; font-weight: 700; padding: 6px 13px; border-radius: 999px; border: 1px solid #bbf7d0; }
.section-capabilities { margin-top: 2.1rem; }
.capabilities-title { font-size: 19px; font-weight: 800; color: #0f2d6b; text-align: center; margin-bottom: 7px; }
.capabilities-sub { text-align: center; font-size: 15px; color: #64748b; margin-bottom: 19px; line-height: 1.5; }
.rcard { border-radius: 15px; padding: 17px; height: 100%; min-height: 176px; box-sizing: border-box; }
.rc-intent { background: #eff6ff; border: 1.5px solid #bfdbfe; }
.rc-conf   { background: #f0fdf4; border: 1.5px solid #bbf7d0; }
.rc-desc   { background: #faf5ff; border: 1.5px solid #e9d5ff; }
.rc-action { background: #fff7ed; border: 1.5px solid #fed7aa; }
.rc-top { display: flex; align-items: center; gap: 7px; margin-bottom: 11px; }
.rc-lbl { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.07em; }
.lbl-i { color: #1d4ed8; } .lbl-c { color: #15803d; } .lbl-d { color: #7c3aed; } .lbl-a { color: #c2410c; }
.rc-main { font-size: 14px; font-weight: 700; color: #1e293b; margin-bottom: 7px; word-break: break-word; font-family: ui-monospace, monospace; }
.rc-sub  { font-size: 14px; color: #475569; line-height: 1.5; font-weight: 600; }
.rc-body { font-size: 13px; color: #64748b; line-height: 1.55; margin-top: 5px; }
.rc-tag  { display: inline-block; margin-top: 11px; font-size: 11px; padding: 5px 11px; border-radius: 20px; font-weight: 700; }
.tag-i { background: #dbeafe; color: #1d4ed8; }
.tag-c { background: #dcfce7; color: #15803d; }
.tag-d { background: #f3e8ff; color: #7c3aed; }
.conf-donut-wrap { display: flex; flex-direction: column; align-items: center; justify-content: flex-start; }
.conf-donut-svg { display: block; margin: 0 auto; }
.bcard { background: white; border-radius: 15px; padding: 17px; border: 1px solid #e2e8f0; }
.bcard-title { font-size: 14px; font-weight: 700; color: #1e293b; margin-bottom: 13px; }
.bar-row { display: flex; align-items: center; gap: 9px; margin-bottom: 8px; }
.bar-lbl { font-size: 12px; color: #475569; width: 178px; flex-shrink: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.bar-wrap { flex: 1; height: 7px; background: #f1f5f9; border-radius: 4px; overflow: hidden; }
.bar-main { height: 100%; border-radius: 3px; background: #2563eb; }
.bar-alt  { height: 100%; border-radius: 3px; background: #cbd5e1; }
.bar-pct  { font-size: 12px; color: #64748b; width: 34px; text-align: right; font-weight: 600; }
.cmp-row { display: flex; align-items: center; justify-content: space-between; padding: 13px; border-radius: 11px; margin-bottom: 9px; }
.cmp-bert-row  { background: #eff6ff; border: 1px solid #bfdbfe; }
.cmp-tfidf-row { background: #f8fafc; border: 1px solid #e2e8f0; }
.cmp-left { display: flex; flex-direction: column; gap: 5px; }
.cmp-name { font-size: 14px; font-weight: 700; }
.cmp-bert-name  { color: #1d4ed8; }
.cmp-tfidf-name { color: #475569; }
.pred-correct { background: #dcfce7; color: #15803d; font-size: 12px; padding: 4px 9px; border-radius: 20px; font-weight: 600; display: inline-block; }
.pred-wrong   { background: #fee2e2; color: #b91c1c; font-size: 12px; padding: 4px 9px; border-radius: 20px; font-weight: 600; display: inline-block; }
.cmp-right { display: flex; gap: 17px; }
.cmp-stat { text-align: center; }
.cmp-val { font-size: 19px; font-weight: 800; }
.cmp-bert-val  { color: #1d4ed8; }
.cmp-tfidf-val { color: #64748b; }
.cmp-lbl { font-size: 11px; color: #94a3b8; }
.style-card { border-radius: 15px; padding: 17px 15px; text-align: center; border: 1px solid transparent; }
.style-formal { background: #ecfdf5; border-color: #a7f3d0; }
.style-casual { background: #eff6ff; border-color: #bfdbfe; }
.style-typo   { background: #fff7ed; border-color: #fed7aa; }
.style-icon  { font-size: 24px; margin-bottom: 9px; }
.style-title { font-size: 14px; font-weight: 700; margin-bottom: 7px; }
.style-ex    { font-size: 13px; color: #475569; font-style: italic; line-height: 1.45; }
.link-btn { display: inline-flex; align-items: center; gap: 7px; background: linear-gradient(180deg, #fb923c, #ea580c); color: white !important; border: none; border-radius: 11px; padding: 9px 17px; font-size: 13px; font-weight: 700; text-decoration: none; margin-top: 13px; box-shadow: 0 2px 8px rgba(234,88,12,0.35); }
.link-btn:hover { filter: brightness(1.05); color: white !important; }
.st-key-input_card .stTextInput > div > div > input { border: 1px solid #e2e8f0 !important; border-radius: 11px !important; padding: 12px 15px !important; font-size: 16px !important; background: #ffffff !important; color: #1e293b !important; min-height: 0 !important; height: 49px !important; box-sizing: border-box !important; }
.st-key-input_card .stTextInput > div > div > input:focus { border-color: #3b82f6 !important; box-shadow: 0 0 0 3px rgba(59,130,246,0.12) !important; }
.st-key-input_card .stTextInput > div > div > input::placeholder { color: #94a3b8 !important; }
.stButton > button[kind="primary"] { background: linear-gradient(180deg, #2563eb, #1d4ed8) !important; color: white !important; border: none !important; border-radius: 11px !important; padding: 0 20px !important; font-size: 16px !important; font-weight: 700 !important; height: 49px !important; min-height: 49px !important; box-shadow: 0 4px 14px rgba(37,99,235,0.35) !important; }
.stButton > button[kind="primary"]:hover { background: linear-gradient(180deg, #1d4ed8, #1e40af) !important; transform: translateY(-1px); }
.block-container { padding-top: 0.45rem !important; padding-bottom: 2.1rem !important; max-width: 1160px; }
div[data-testid="stVerticalBlock"] > div { gap: 0.45rem; }
footer { visibility: hidden; }
.app-footer { text-align: center; color: #94a3b8; font-size: 14px; margin-top: 2.1rem; padding-top: 1.05rem; }
</style>
""", unsafe_allow_html=True)

# --- navbar ---
st.markdown("""
<div class="navbar">
  <div class="nav-left">
    <div class="nav-logo">SGK</div>
    <div class="nav-title">SGK Akıllı Niyet Tespit Sistemi</div>
  </div>
  <div class="nav-right">
    <span class="nav-pill-gray">👤 BERTurk Modeli</span>
    <span class="nav-pill-blue">🛡️ Yüksek Doğruluk</span>
  </div>
</div>
""", unsafe_allow_html=True)

# --- hero ---
st.markdown("""
<div class="hero">
  <h1>SGK Akıllı Niyet Tespit Sistemi</h1>
  <p class="hero-lead">Kullanıcının doğal Türkçe ifadelerini anlayarak ilgili SGK işlem niyetini tahmin eder.</p>
</div>
""", unsafe_allow_html=True)

# --- model yükle ---
tokenizer, bert_model, id2label, baseline_model, vectorizer = load_models()

if "user_input_area" not in st.session_state:
    st.session_state.user_input_area = ""

# --- giriş kartı ---
with st.container(border=True, key="input_card"):
    col_in, col_btn = st.columns([5, 1], vertical_alignment="center")
    with col_in:
        user_input = st.text_input(
            "Sorgu",
            max_chars=MAX_INPUT_CHARS,
            placeholder="Örn. Sigortam aktif mi öğrenmek istiyorum…",
            key="user_input_area",
            label_visibility="collapsed",
        )
    with col_btn:
        predict_clicked = st.button("✨ Tahmin Et", type="primary", use_container_width=True)

    with st.container(key="examples_chips_row"):
        chip_list = EXAMPLE_CHIP_ROW
        n_chip = len(chip_list)
        ex_cols = st.columns([4] + [6] * n_chip, vertical_alignment="center", gap="small")
        with ex_cols[0]:
            st.markdown(
                '<div class="examples-inline-label">'
                '<svg class="ex-sparkle-svg" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
                '<path fill="#2563eb" d="M12 2.5L13.1 8.9L19.5 10L13.1 11.1L12 17.5L10.9 11.1L4.5 10L10.9 8.9L12 2.5Z"/>'
                "</svg>"
                '<span class="ex-label-text">Örnek sorgular:</span>'
                "</div>",
                unsafe_allow_html=True,
            )
        for i, (label, fill_text) in enumerate(chip_list):
            with ex_cols[i + 1]:
                st.button(label, use_container_width=True, key=f"ex_chip_{i}", on_click=_set_example_query, args=(fill_text,))

# --- tahmin ---
if predict_clicked and (user_input or "").strip():
    with st.spinner("Model analiz ediyor..."):
        top_intent, top_conf, all_results = predict_bert(
            user_input, tokenizer, bert_model, id2label
        )
        baseline_pred = predict_baseline(user_input, baseline_model, vectorizer)

    # --- YENİ: status kontrolü ---
    status = get_prediction_status(user_input, top_conf, all_results)

    if status == "out_of_domain":
        st.markdown("""
        <div style="background:#fef2f2;border:1.5px solid #fecaca;border-radius:14px;padding:20px 24px;margin-top:16px;display:flex;align-items:flex-start;gap:14px;">
          <span style="font-size:28px;">🚫</span>
          <div>
            <div style="font-size:15px;font-weight:700;color:#b91c1c;margin-bottom:6px;">Bu sorgu SGK işlemleriyle ilgili görünmüyor</div>
            <div style="font-size:13px;color:#7f1d1d;">Lütfen SGK ile ilgili bir soru yazın. Örneğin: prim günlerinizi, sigorta durumunuzu veya emeklilik şartlarınızı sorabilirsiniz.</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    if status == "uncertain":
        st.markdown("""
        <div style="background:#fffbeb;border:1.5px solid #fde68a;border-radius:14px;padding:20px 24px;margin-top:16px;display:flex;align-items:flex-start;gap:14px;">
          <span style="font-size:28px;">🤔</span>
          <div>
            <div style="font-size:15px;font-weight:700;color:#92400e;margin-bottom:6px;">Ne demek istediğinizi tam anlayamadım</div>
            <div style="font-size:13px;color:#78350f;">Sorunuzu biraz daha açık yazabilir misiniz? Örneğin: "SGK borcum var mı?" veya "Emeklilik şartlarım neler?" gibi.</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # --- normal akış ---
    meta     = INTENT_META[top_intent]
    conf_pct = int(round(top_conf * 100))
    conf_lbl = conf_label(top_conf)

    with st.container(border=True, key="result_panel"):
        st.markdown("""
        <div class="result-head" style="margin-bottom:15px;">
          <div class="result-head-left"><span>📊</span> Tahmin Sonucu</div>
          <span class="badge-instant">Anında Tahmin</span>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        donut_html = confidence_donut_svg(conf_pct)

        with c1:
            st.markdown(f"""
        <div class="rcard rc-intent">
          <div class="rc-top"><span class="rc-lbl lbl-i">Tahmin Edilen Niyet</span></div>
          <div class="rc-main">{top_intent}</div>
          <div class="rc-sub">{meta["title"]}</div>
          <span class="rc-tag tag-i">Ana Niyet</span>
        </div>""", unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
        <div class="rcard rc-conf">
          <div class="rc-top"><span class="rc-lbl lbl-c">Güven Skoru</span></div>
          <div class="conf-donut-wrap">{donut_html}</div>
          <span class="rc-tag tag-c" style="display:block;text-align:center;margin-top:10px;">{conf_lbl}</span>
        </div>""", unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
        <div class="rcard rc-desc">
          <div class="rc-top"><span class="rc-lbl lbl-d">Açıklama</span></div>
          <div class="rc-body">{meta["description"]}</div>
          <span class="rc-tag tag-d">Niyet Açıklaması</span>
        </div>""", unsafe_allow_html=True)

        with c4:
            st.markdown(f"""
        <div class="rcard rc-action">
          <div class="rc-top"><span class="rc-lbl lbl-a">Önerilen İşlem</span></div>
          <div class="rc-body">{meta["action"]}</div>
          <a class="link-btn" href="{meta["link"]}" target="_blank" rel="noopener noreferrer">e-Devlet'e Git ↗</a>
        </div>""", unsafe_allow_html=True)

    with st.expander("Alternatif tahminler ve model karşılaştırması", expanded=True):
        col_bars, col_cmp = st.columns(2)

        with col_bars:
            bars_html = '<div class="bcard"><div class="bcard-title">📊 Alternatif tahminler</div>'
            for intent, prob in all_results:
                pct = int(round(prob * 100))
                css = "bar-main" if intent == top_intent else "bar-alt"
                m = INTENT_META.get(intent, {})
                bars_html += f"""
            <div class="bar-row">
              <span class="bar-lbl">{m.get("icon", "")} {intent}</span>
              <div class="bar-wrap"><div class="{css}" style="width:{pct}%"></div></div>
              <span class="bar-pct">{pct}%</span>
            </div>"""
            bars_html += "</div>"
            st.markdown(bars_html, unsafe_allow_html=True)

        with col_cmp:
            tfidf_correct = baseline_pred == top_intent
            tfidf_css = "pred-correct" if tfidf_correct else "pred-wrong"
            tfidf_icon = "✓" if tfidf_correct else "✗"
            st.markdown(f"""
        <div class="bcard">
          <div class="bcard-title">⚖️ Model karşılaştırması</div>
          <div class="cmp-row cmp-bert-row">
            <div class="cmp-left">
              <span class="cmp-name cmp-bert-name">🔵 BERTurk Fine-tuned</span>
              <span class="pred-correct">✓ {top_intent}</span>
            </div>
            <div class="cmp-right">
              <div class="cmp-stat"><div class="cmp-val cmp-bert-val">%90.7</div><div class="cmp-lbl">Accuracy</div></div>
              <div class="cmp-stat"><div class="cmp-val cmp-bert-val">0.90</div><div class="cmp-lbl">F1</div></div>
            </div>
          </div>
          <div class="cmp-row cmp-tfidf-row">
            <div class="cmp-left">
              <span class="cmp-name cmp-tfidf-name">⚪ TF-IDF + LR</span>
              <span class="{tfidf_css}">{tfidf_icon} {baseline_pred}</span>
            </div>
            <div class="cmp-right">
              <div class="cmp-stat"><div class="cmp-val cmp-tfidf-val">%81.5</div><div class="cmp-lbl">Accuracy</div></div>
              <div class="cmp-stat"><div class="cmp-val cmp-tfidf-val">0.81</div><div class="cmp-lbl">F1</div></div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

elif predict_clicked and not (user_input or "").strip():
    st.warning("Lütfen bir sorgu girin.")

# --- model neleri anlayabilir ---
st.markdown('<div class="section-capabilities">', unsafe_allow_html=True)
st.markdown('<div class="capabilities-title">Model Neleri Anlayabilir?</div>', unsafe_allow_html=True)
st.markdown('<div class="capabilities-sub">Günlük, bozuk veya resmi fark etmeksizin doğal dildeki sorguları anlayabilir.</div>', unsafe_allow_html=True)
s1, s2, s3 = st.columns(3)
with s1:
    st.markdown("""
    <div class="style-card style-formal">
      <div class="style-icon">📝</div>
      <div class="style-title" style="color:#15803d;">Resmi Dil</div>
      <div class="style-ex">"Sigorta hizmet dökümümü nasıl alabilirim?"</div>
    </div>""", unsafe_allow_html=True)
with s2:
    st.markdown("""
    <div class="style-card style-casual">
      <div class="style-icon">💬</div>
      <div class="style-title" style="color:#1d4ed8;">Günlük Dil</div>
      <div class="style-ex">"Sigortam gözüküyo mu?"</div>
    </div>""", unsafe_allow_html=True)
with s3:
    st.markdown("""
    <div class="style-card style-typo">
      <div class="style-icon">⌨️</div>
      <div class="style-title" style="color:#c2410c;">Bozuk Yazım</div>
      <div class="style-ex">"sgk borcum varmı ya"</div>
    </div>""", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="app-footer">© 2024 SGK Akıllı Niyet Tespit Sistemi • BERTurk ile Güçlendirilmiştir</div>', unsafe_allow_html=True)
