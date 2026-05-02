# Kurdish NLP Pipeline

A transformer-based NLP pipeline for Kurdish, built with spaCy and XLM-RoBERTa.
This project covers both Sorani (Central Kurdish) and Kurmanji (Northern Kurdish) dialects.

---

## Results

### NER — Named Entity Recognition (Sorani Kurdish)

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
| Custom annotated corpus | Sorani | NER training (22,800 sentences) |
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
