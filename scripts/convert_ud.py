"""
Convert Universal Dependencies CoNLL-U files to spaCy binary format.

Usage:
    python scripts/convert_ud.py
"""

import spacy
from spacy.tokens import Doc, DocBin
from conllu import parse_incr


nlp = spacy.blank("xx")


def convert(input_path: str, output_path: str) -> None:
    """Convert a CoNLL-U file to a spaCy DocBin (.spacy) file."""
    db = DocBin()

    with open(input_path, "r", encoding="utf-8") as f:
        for sent in parse_incr(f):
            words = [token["form"] for token in sent]

            doc = Doc(nlp.vocab, words=words)

            for i, token in enumerate(sent):
                if token["upostag"]:
                    doc[i].pos_ = token["upostag"]

                if token["deprel"]:
                    doc[i].dep_ = token["deprel"]

                if token["lemma"] and token["lemma"] != "_":
                    doc[i].lemma_ = token["lemma"]

            db.add(doc)

    db.to_disk(output_path)
    print(f"Saved {len(db)} docs to {output_path}")


if __name__ == "__main__":
    import os

    os.makedirs("data/processed", exist_ok=True)

    convert("data/raw/train.conllu", "data/processed/train.spacy")
    convert("data/raw/dev.conllu", "data/processed/dev.spacy")
