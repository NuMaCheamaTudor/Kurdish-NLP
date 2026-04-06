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
                # POS
                if token["upostag"] and token["upostag"] != "_":
                    doc[i].pos_ = token["upostag"]

                # TAG
                if token["xpostag"] and token["xpostag"] != "_":
                    doc[i].tag_ = token["xpostag"]
                elif token["upostag"] and token["upostag"] != "_":
                    doc[i].tag_ = token["upostag"]

            # 🔥 IMPORTANT: HEAD + DEP
            for i, token in enumerate(sent):
                if token["head"] and token["head"] != "_":
                    head = int(token["head"])
                    if head == 0:
                        doc[i].head = doc[i]  # ROOT
                    else:
                        doc[i].head = doc[head - 1]

                if token["deprel"] and token["deprel"] != "_":
                    doc[i].dep_ = token["deprel"]

            if len(sent) > 0:
                print(f"First sentence tokens:")
                for i in range(min(5, len(sent))):
                    print(f"  {doc[i].text}: pos={doc[i].pos_}, dep={doc[i].dep_}")

            db.add(doc)

    db.to_disk(output_path)
    print(f"Saved {len(db)} docs to {output_path}")

    # Test load
    db2 = DocBin().from_disk(output_path)
    docs = list(db2.get_docs(nlp.vocab))
    print(f"Loaded {len(docs)} docs")
    for doc in docs[:1]:
        print("Loaded first doc tokens:")
        for token in doc[:5]:
            print(f"  {token.text}: pos={token.pos_}, dep={token.dep_}")


if __name__ == "__main__":
    import os

    os.makedirs("data/processed", exist_ok=True)

    convert("data/raw/kmr_kurmanji-ud-train.conllu", "data/processed/train.spacy")
    convert("data/raw/kmr_kurmanji-ud-test.conllu", "data/processed/dev.spacy")
