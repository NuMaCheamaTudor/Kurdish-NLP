# Kurdish spaCy Model

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
