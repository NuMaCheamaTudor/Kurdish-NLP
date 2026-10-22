# Kurdish NLP Pipeline

A transformer-based NLP pipeline for Kurdish, built with spaCy and XLM-RoBERTa.
This project covers both Sorani (Central Kurdish) and Kurmanji (Northern Kurdish) dialects.

---

## Results

### NER — Named Entity Recognition (Sorani Kurdish)

These historical NER scores are **not a trustworthy baseline**: the saved V1
configuration uses `ner.spacy` for both training and development. They are kept
here only as a historical record and must not be compared with V2 results.
The [Sorani NER V2 foundation](docs/sorani-ner-v2.md) provides an independently
audited corpus, frozen leakage-aware splits, and a validated real XLM-R smoke
test. A full V2 model and held-out test results are not available yet.

| Metric | Score |
|--------|-------|
| F1 | **88.17** |
| Precision | 90.33 |
| Recall | 86.10 |


**Per-label scores:**

| Label | F1 |
|-------|----|
| DAY | 95.61 |
| COUNTRY | 94.83 |
| CURRENCY | 93.99 |
| LOC | 91.27 |
| SPORT | 90.26 |
| PER | 89.35 |
| MISC | 86.32 |
| ORG | 77.86 |
| POSITION | 62.76 |

---

### POS — POS Tagging + Morphology + Dependency Parsing (Kurmanji Kurdish)

| Metric | Score |
|--------|-------|
| POS Accuracy | **93.55%** |
| Morphology | **69.76%** |
| Dependency LAS | **69.01%** |
| Sentence Detection F1 | **93.88%** |

**Per-feature morphology scores:**

| Feature | F1 |
|---------|----|
| VerbForm | 97.28 |
| Reflex | 97.96 |
| AdpType | 95.95 |
| Definite | 88.51 |
| Number | 87.94 |
| Person | 90.68 |
| Case | 79.25 |

---

## Models

| Model | Dialect | Task | Download |
|-------|---------|------|----------|
| kurdish-ner-v1 | Sorani | NER (16 labels) | Coming soon |
| kurdish-pos-v1 | Kurmanji | POS + Morphology + Dependencies | Coming soon |

---

## Setup

```bash
git clone https://github.com/NuMaCheamaTudor/Kurdish-NLP.git
cd Kurdish-NLP
pip install -r requirements.txt
```

---

## Usage

### NER

```python
import spacy

nlp = spacy.load("kurdish-ner-v1")
doc = nlp("ئاژانسی ناسا پلانی هەیە")

for ent in doc.ents:
    print(ent.text, ent.label_)
```

### POS + Morphology + Dependency Parsing

```python
import spacy

nlp = spacy.load("kurdish-pos-v1")
doc = nlp("Ev kitêb baş e")

for token in doc:
    print(token.text, token.pos_, token.lemma_, token.dep_)
```

---

## Project Structure

```
Kurdish-NLP/
├── notebooks/
│   ├── kurdish_ner.ipynb           # NER training notebook
│   ├── kurdish_pos.ipynb           # POS training notebook
│   └── kurdish_lemmatizer.ipynb    # Lemmatizer notebook
├── configs/
│   ├── trf_ner.cfg                 # NER config
│   └── trf.cfg                     # POS config
├── scripts/
│   ├── convert_ner.py              # BIO to spaCy conversion
│   ├── convert_ud.py               # CoNLL-U to spaCy conversion
│   ├── train.py                    # Training script
│   └── evaluate.py                 # Evaluation script
├── data/
│   ├── raw/                        # Source datasets (not tracked)
│   └── processed/                  # Converted .spacy files (not tracked)
├── kmr_pipeline/                   # Pipeline package
├── requirements.txt
└── README.md
```

---

## Data Sources

| Dataset | Dialect | Used for |
|---------|---------|----------|
| Custom annotated corpus | Sorani | NER source: 22,800 BIO segments; original document identities unknown |
| UD Northern Kurdish-Kurmanji | Kurmanji | POS/Morphology/Dependency (734 sentences) |

---


## Tech Stack

| Tool | Version | Purpose |
|------|---------|---------|
| spaCy | 3.8.x | NLP framework |
| spacy-transformers | 1.4.0 | Transformer integration |
| XLM-RoBERTa base | — | Multilingual encoder |
| transformers | 4.46.3 | HuggingFace model loading |
| PyTorch | — | Training backend |

---

## Language identification foundation

The repository now includes a backend-independent foundation for future detection
of `ckb`, `kmr`, `sdh`, `ar`, `fa`, `tr`, and `en`, with `und` as a conservative
runtime abstention result. It provides typed results, safe normalization, canonical
JSONL validation/conversion tooling, and an optional fastText adapter.
Offline-audited acquisition/import pipelines are available for PARME, Tatoeba,
and six Wikimedia editions. A reproducible builder
(`kurdish-langid-data build dataset-v1`) combines their verified exports into a
deduplicated, group-aware train/dev/test release with UD v2.18 reserved for
external evaluation; generated corpora stay git-ignored and only the
configuration and output-hash lock are tracked.

No production-quality detector or trained Language ID model is included yet. Real,
balanced datasets, held-out evaluation, and confidence calibration are still
required. See [the Language ID documentation](docs/language-identification.md).

---

## Kurdish NLP Studio (local web app)

A local workspace for language identification: a FastAPI service (`apps/api`)
in front of the trained baseline-v1 classifiers, and a React app (`apps/web`).
Named entities and grammar appear as unavailable until a complete spaCy
pipeline is installed; nothing is simulated.

### Model files

The trained models are git-ignored and **not publicly hosted yet**, so a fresh
clone does not include them. The API expects the frozen experiment directory:

```
artifacts/langid/experiments/baseline-v1/
  selection.lock.json, experiment_manifest.json
  tfidf_lr/  inference.json, model.joblib, model.joblib.meta.json, evaluation*.json
  fasttext/  inference.json, model.bin, evaluation*.json
```

Copy it from a machine that has it, or point `KNLP_ARTIFACTS_DIR` at another
copy. Regenerating it means rebuilding Dataset V1 and rerunning Experiment 01
(see the Language ID documentation). Every artifact is hash-checked against its
inference spec and the selection lock before loading. The TF-IDF pickle loads
only under the scikit-learn version that saved it (1.9.1). A missing or
mismatched model is reported as unavailable; the app keeps running.

### Run

```bash
# Backend (Python 3.11+), from the repository root
python -m venv .venv && source .venv/bin/activate
pip install -e '.[experiments]'
pip install -r apps/api/requirements.txt
uvicorn app.main:app --app-dir apps/api --port 8000

# Frontend (Node 22.12+), in a second terminal
cd apps/web
npm install
npm run dev            # http://localhost:5173, proxies /api to port 8000
```

Optional settings live in `apps/api/.env.example` and `apps/web/.env.example`
(artifact path, default model, text limit, CORS origins, API URL).

### Test

```bash
cd apps/api && pytest                  # API, against the real models when present
cd apps/web && npm test                # components and flows (Vitest)
cd apps/web && npx playwright install chromium && npm run e2e   # end to end, starts both servers
```

---

## Citation

If you use this work please cite:

```bibtex
@misc{kurdish-nlp-2026,
  author    = {Tudor},
  title     = {Kurdish NLP Pipeline — Transformer-based NER and POS for Kurdish},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/NuMaCheamaTudor/Kurdish-NLP}
}
```

---

## License

This project is licensed under the MIT License.
The UD Northern Kurdish-Kurmanji dataset is licensed under CC BY-SA 4.0.


The current [Colab notebook](notebooks/sorani_ner_v2_colab.ipynb) includes the T4
AMP accumulation/overflow correction and gradient checkpointing. Prepare its
matching code zip with `scripts/prepare_ner_colab.py` (default output:
`artifacts/ner/colab-stress-fix/sorani-ner-v2-code.zip`). Use a fresh run ID and
repeat all GPU preflights. See the [stress failure analysis](docs/sorani-ner-v2-colab-stress-fix.md);
T4 training capacity still requires Colab verification.
