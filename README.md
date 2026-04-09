# Sentiment Analysis — COPOM & FOMC Minutes

NLP-based sentiment scoring of central bank communications (Brazil's COPOM and the US FOMC) to construct a monetary policy directional indicator — the **Copômetro** — and a **FOMC-Meter**.

Uses sentence embeddings + Random Forest to classify each relevant paragraph as hawkish, dovish, or neutral, then aggregates scores into a continuous directional signal validated against the Selic rate and 2-year swap curves.

---

## What it does

1. **Downloads** COPOM and FOMC meeting minutes as PDFs from official sources (BCB and FederalReserve.gov)
2. **Converts** PDFs to cleaned plain text, stripping headers, dates, and page numbers
3. **Classifies** each paragraph using a fine-tuned sentence-transformer + Random Forest classifier trained on a labeled dataset of hawkish/dovish/neutral excerpts
4. **Aggregates** scores per meeting into a smoothed continuous signal
5. **Validates** the signal against actual rate changes (Selic delta, 2Y swap) using HP-filter surprise extraction, cross-validation, and rolling-window backtests

---

## Results (COPOM model)

- Directional accuracy (non-neutral signals): ~70%+
- Rolling-window validation across 5 periods (2012–2020)
- Out-of-bag R² on Random Forest regressor
- Residual analysis and learning curve diagnostics included

---

## Project Structure

```
├── Analise_deSentimento_COPOMv6.py   # Main COPOM model (training + validation + plots)
├── Analise_deSentimento_FEDv6.py     # FOMC-Meter equivalent
├── extract_COPOM.py                  # Selenium scraper: downloads COPOM PDFs from BCB
├── FED_DOWNLOAD.PY                   # Scraper: downloads FOMC minutes from FederalReserve.gov
├── FED_TXT_TO_PDF.PY                 # Converts FOMC PDFs to clean .txt
├── atas_copom/                       # COPOM minutes (PDF + TXT — populate via extract_COPOM.py)
├── atas_fomc/                        # FOMC minutes (PDF + TXT — populate via FED_DOWNLOAD.PY)
├── Base_SELIC.xlsx                   # Selic rate history (columns: data, delta_selic_t1 verdadeiro)
├── Base_2Y.xlsx                      # 2-year swap rate history (columns: date, swap2y)
└── copom_additional_refined_902.xlsx # Labeled guidance dataset (texto, tipo: hawkish/dovish/neutral)
```

---

## Setup

```bash
pip install -r requirements.txt
python -m nltk.downloader punkt
```

### Data Requirements

You need to provide the following files (not included in the repo):

| File | Source |
|------|--------|
| `Base_SELIC.xlsx` | BCB — Selic rate history |
| `Base_2Y.xlsx` | Bloomberg / B3 — DI x IPCA 2-year swap |
| `copom_additional_refined_902.xlsx` | Manually labeled COPOM guidance excerpts |

Meeting minutes can be downloaded automatically:

```bash
# Download COPOM PDFs from BCB
python extract_COPOM.py

# Convert COPOM PDFs to TXT (if not using Selenium version)
# (integrated in extract_COPOM.py)

# Download FOMC minutes
python FED_DOWNLOAD.PY

# Convert FOMC PDFs to TXT
python FED_TXT_TO_PDF.PY
```

---

## Running the models

```bash
# COPOM Copômetro
python Analise_deSentimento_COPOMv6.py

# FOMC-Meter
python Analise_deSentimento_FEDv6.py
```

Outputs are saved to `outputs/`:
- `copometro_rf_fullswap_selic.png` — Signal vs Swap2y and Selic
- `rolling_window_*.png` — Rolling validation plots
- `learning_curve_rf.png` — Learning curve diagnostic
- `importancia_features_copometro_rf.png` — Feature importances
- `rolling_window_metrics.xlsx` — Numeric validation summary

---

## Methodology

### Sentence Embeddings
Uses `paraphrase-MiniLM-L6-v2` (SentenceTransformers) to encode COPOM paragraph text into 384-dimensional vectors.

### Guidance Classification
Random Forest classifier (`n_estimators=200`, OOB scoring) trained on ~900 labeled COPOM excerpts.
Labels: `hawkish` (+1), `neutral` (0), `dovish` (−1).

### HP-Filter Surprise
Hodrick-Prescott filter (`λ=2500`) applied recursively to extract the cyclical component (policy surprise) from the smoothed sentiment signal.

### Validation
- Train/test split (70/30, no shuffle — respects time order)
- 5-fold cross-validation on training set
- Rolling-window validation: train through year Y, test on Y+1 through end

---

## License

MIT
