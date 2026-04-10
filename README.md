  ---
  # SGK Intent Detector

  A Turkish NLP project that classifies user queries related to the Turkish Social Security Institution (SGK) into
  predefined intent categories. The project compares a classical machine learning baseline against a fine-tuned
  BERT-based transformer model, and includes an interactive Streamlit demo app.

  ---

  ## Overview

  Users often struggle to navigate SGK's services. This project builds an intent detection system that identifies what a
   user is asking about — so chatbots or help systems can route them to the right service automatically.

  The dataset contains **873 labeled Turkish sentences** covering **10 SGK-specific intents** plus an `out_of_scope`
  class (11 classes total).

  ---

  ## Intents

  | Intent | Description |
  |---|---|
  | `insurance_status_query` | Checking active insurance registration |
  | `premium_days_query` | Querying total premium days |
  | `premium_payment_query` | Questions about premium payments |
  | `sgk_debt_query` | SGK / GSS debt inquiry |
  | `retirement_query` | Retirement eligibility and date |
  | `employment_status_query` | Current employment status |
  | `health_coverage_query` | Health insurance coverage |
  | `service_record_query` | Service record / work history |
  | `registration_document_query` | Registration documents |
  | `general_info_query` | General SGK information |
  | `out_of_scope` | Queries unrelated to SGK |

  ---

  ## Models & Results

  | Model | Accuracy | Macro F1 |
  |---|---|---|
  | TF-IDF + Logistic Regression | 79.65% | 73.83% |
  | BERTurk (`dbmdz/bert-base-turkish-cased`) | **92.04%** | **90.95%** |

  The BERTurk model was fine-tuned for 8 epochs on the cleaned dataset and significantly outperforms the baseline,
  especially on semantically similar intent pairs.

  ---

  ## Project Structure

  sgk-intent-detector/
  ├── data/
  │   └── sgk_dataset_improved_v4.xlsx   # Final labeled dataset (873 rows)
  ├── models/
  │   ├── baseline_model.pkl             # Trained TF-IDF + LR model
  │   ├── berturk_best.pt                # Best BERTurk checkpoint
  │   ├── tfidf_vectorizer.pkl
  │   ├── label2id.pkl
  │   └── id2label.pkl
  ├── outputs/
  │   ├── evaluation_summary.json        # Accuracy & F1 scores
  │   ├── baseline_confusion_matrix.png
  │   └── berturk_confusion_matrix.png
  ├── scripts/
  │   └── evaluate_models.py
  ├── 00_preprocess.py                   # Data cleaning & preprocessing
  ├── 01_baseline.py                     # TF-IDF + Logistic Regression
  ├── 02_berturk.py                      # BERTurk fine-tuning
  ├── 03_app.py                          # Streamlit demo app
  └── requirements.txt

  ---

  ## Setup

  **1. Clone the repository and create a virtual environment:**

  ```bash
  git clone https://github.com/your-username/sgk-intent-detector.git
  cd sgk-intent-detector
  python -m venv .venv
  source .venv/bin/activate  # Windows: .venv\Scripts\activate

  2. Install dependencies:

  pip install -r requirements.txt

  ---
  Usage

  Run the full pipeline

  # 1. Preprocess the data
  python 00_preprocess.py

  # 2. Train the baseline model
  python 01_baseline.py

  # 3. Fine-tune BERTurk
  python 02_berturk.py

  Launch the demo app

  streamlit run 03_app.py

  The app lets you type a Turkish SGK query and see the predicted intent, confidence score, and a direct link to the
  relevant e-Government service.

  ---
  Tech Stack

  - Python 3.10+
  - scikit-learn — TF-IDF vectorization, Logistic Regression
  - Transformers (HuggingFace) — BERTurk fine-tuning
  - PyTorch — model training
  - Streamlit — interactive demo
  - pandas / numpy / matplotlib / seaborn — data processing and visualization

  ---
  Dataset Notes

  - Text was manually authored and augmented to cover realistic SGK query patterns in Turkish.
  - Two text variants are maintained: text_clean (lowercased, punctuation removed — used by baseline) and text_bert
  (original casing preserved — used by BERTurk).
  - The dataset went through 4 improvement iterations to reduce class overlap and increase coverage.
