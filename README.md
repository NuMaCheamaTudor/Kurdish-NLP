# Kurdish spaCy Model


### POS v1 — POS Tagging + Morphology + Dependency Parsing (Kurmanji Kurdish)
- Model: XLM-RoBERTa base
- Dataset: 587 sentences (UD Northern Kurdish-Kurmanji treebank)
- POS Accuracy: 93.55%
- Morphology Accuracy: 69.76%
- Dependency LAS: 69.01%

---
## Data Sources

- NER: Custom annotated Sorani Kurdish corpus
- POS: [UD Northern Kurdish-Kurmanji](https://github.com/UniversalDependencies/UD_Northern_Kurdish-Kurmanji) (CC BY-SA 4.0)

## Setup

```bash
pip install -r requirements.txt
```

## Steps

1. Convert NER dataset

```bash
python scripts/convert_ner.py
```

2. Clean corpus

```bash
python scripts/clean_corpus.py
```

3. Train model

```bash
python scripts/train.py
```

4. Evaluate

```bash
python scripts/evaluate.py
```
