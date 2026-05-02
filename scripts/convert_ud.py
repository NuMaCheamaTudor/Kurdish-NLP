import spacy
from spacy.tokens import Doc, DocBin
from conllu import parse_incr

nlp = spacy.blank("xx")

def convert(input_path, output_path):
    db = DocBin()

    with open(input_path, "r", encoding="utf-8") as f:
        for sent in parse_incr(f):

            # ❌ skip multiword tokens
            tokens = [t for t in sent if isinstance(t["id"], int)]

            words = [t["form"] for t in tokens]

            doc = Doc(nlp.vocab, words=words)

            # ✅ sentence boundary
            doc[0].is_sent_start = True

            # POS + TAG
            for i, token in enumerate(tokens):
                doc[i].pos_ = token["upostag"]
                doc[i].tag_ = token["xpostag"] or token["upostag"]

            # HEAD + DEP
            for i, token in enumerate(tokens):
                head = token["head"]

                if head == 0:
                    doc[i].head = doc[i]  # ROOT
                else:
                    doc[i].head = doc[head - 1]

                doc[i].dep_ = token["deprel"]

            db.add(doc)

    db.to_disk(output_path)


if __name__ == "__main__":
    convert("data/raw/train.conllu", "data/processed/train.spacy")
    convert("data/raw/dev.conllu", "data/processed/dev.spacy")