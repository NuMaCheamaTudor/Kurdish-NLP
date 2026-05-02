import subprocess


subprocess.run([
    "python", "-m", "spacy", "train",
    "configs/trf_ner.cfg",
    "--output", "output",
    "--paths.train", "data/processed/ner.spacy",
    "--paths.dev", "data/processed/ner.spacy",
])
