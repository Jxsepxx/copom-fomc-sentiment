# COPOM & FOMC Minutes — Monetary Policy Tone

Classifies paragraphs of Brazil's COPOM minutes (and the US FOMC minutes) as **hawkish**, **neutral** or **dovish**, aggregates them into a per-meeting tone index (the *Copômetro*), and tests whether that index leads the Selic rate and the 2-year rate.

> **Status: research prototype.** The current validation script (`Analise_deSentimento_COPOMv6.py`) has look-ahead issues, listed under [Known limitations](#known-limitations). Its accuracy figures should **not** be read as out-of-sample performance. A corrected walk-forward version (v7) is in progress.

---

## Pipeline

1. **Collect** — download COPOM minutes (BCB) and FOMC minutes (federalreserve.gov) as PDF.
2. **Parse** — extract numbered paragraphs to plain text; drop paragraphs repeated verbatim from the previous meeting.
3. **Filter** — keep paragraphs that mention policy guidance (keywords such as *monetary policy*, *policy rate*, *forward guidance*).
4. **Classify** — encode each paragraph with `paraphrase-MiniLM-L6-v2` (pretrained, not fine-tuned) and classify it with a Random Forest trained on ~900 manually labeled excerpts (hawkish = +1, neutral = 0, dovish = −1).
5. **Aggregate** — average paragraph scores per meeting into a tone index.
6. **Extract surprise** — one-sided (recursive) Hodrick–Prescott filter, λ = 2500, using only data available up to each meeting.
7. **Test** — relate the lagged surprise to the next change in Selic and to the 2-year rate.

## Known limitations

These are the reasons the current numbers are not trustworthy. v7 fixes each one.

| Issue | Where | Effect |
| --- | --- | --- |
| Reported directional accuracy is computed on a model fitted to the full sample | final block, `rf_full` | In-sample number |
| Centered rolling mean (`rolling(2, center=True)`) | tone smoothing | Uses the next meeting to smooth the current one |
| Scalers and the 90th-percentile shock threshold fit on the full history | feature construction | Future information leaks into features |
| Model trained on Selic, accuracy measured on 2-year rate direction | evaluation | Target mismatch |
| "Rolling window" is expanding, with overlapping test sets | validation loop | Overstates stability |
| Classifier evaluated only by out-of-bag score; labeled excerpts may come from the same minutes later scored | classification | Possible label leakage |

What already works as intended: the HP filter is one-sided, the train/test split respects time order, and repeated boilerplate paragraphs are removed.

## Roadmap (v7)

- [ ] Walk-forward evaluation: every scaler, threshold and model fitted only on data before each meeting
- [ ] No centered smoothing
- [ ] Classifier evaluated on minutes held out by date
- [ ] Naive baseline (repeat the last decision) and comparison against it
- [ ] Package layout (`src/`, `tests/`, `pyproject.toml`) and a small sample dataset so the pipeline runs end to end
- [ ] Results reported with the evaluation window and number of meetings

## Repository structure

```
├── Analise_deSentimento_COPOMv6.py   # COPOM model: training, validation, plots (v6, see limitations)
├── Analise_deSentimento_FEDv6.py     # FOMC equivalent
├── extract_COPOM.py                  # Downloads COPOM minutes from BCB (Selenium)
├── Modelo.py                         # Parses COPOM PDFs into numbered paragraphs
├── FED_DOWNLOAD.PY                   # Downloads FOMC minutes
├── FED_TXT_TO_PDF.PY                 # Converts FOMC PDFs to text (name to be fixed in v7)
└── requirements.txt
```

## Data

Not included in the repository.

| File | Content | Source |
| --- | --- | --- |
| `Base_SELIC.xlsx` | Selic decisions (`data`, `delta_selic_t1 verdadeiro`) | BCB |
| `Base_2Y.xlsx` | 2-year rate (`date`, `swap2y`) | B3 / Bloomberg |
| `copom_additional_refined_902.xlsx` | ~900 labeled excerpts (`texto`, `tipo`) | Manual labeling |

## Running

```bash
pip install -r requirements.txt
python -m nltk.downloader punkt
python extract_COPOM.py
python Analise_deSentimento_COPOMv6.py
```

## License

MIT — see `LICENSE`.
