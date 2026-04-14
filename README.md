# Kurdish spaCy Model (Sorani) - University Project

Hey there! This is my university project where I'm having a go at building a spaCy NLP pipeline for the Kurdish Sorani dialect. It's been a cool learning experience diving into natural language processing and exploring how it works with a language that's not as mainstream in tech.

## Project Scope

The scope here is pretty straightforward – I'm focusing on the core NLP tasks to get a feel for how spaCy handles language processing. We're talking tokenization, POS tagging, morphological analysis, dependency parsing, and lemmatization. This isn't meant to be a super polished, production-ready model, but more of a hands-on way to understand NLP concepts and apply them to Kurmanji Kurdish. It's all about the journey of learning, experimenting, and seeing what works (and what doesn't)!

## Features

- Tokenization
- POS Tagging
- Morphological Analysis
- Dependency Parsing
- Lemmatization

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python scripts/train.py
```

## Evaluate

```bash
python scripts/evaluate.py
```

## Workflow (spaCy project)

```bash
# Convert UD CoNLL-U data to .spacy format
python scripts/convert_ud.py

# Or use spaCy project commands
python -m spacy project run convert
python -m spacy project run train
```

## Data

Place your `.conllu` files in `data/raw/`:
- `data/raw/train.conllu`
- `data/raw/dev.conllu`

Processed `.spacy` files will be saved to `data/processed/`.

## Output

Trained models are saved to `output/model-best` and `output/model-last`.
