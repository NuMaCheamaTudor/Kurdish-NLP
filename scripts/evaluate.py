import subprocess


subprocess.run([
    "python", "-m", "spacy", "evaluate",
    "output/model-best",
    "data/processed/ner.spacy",
])
