import spacy
from spacy.tokens import DocBin


def convert_bio_to_spacy(input_file, output_file):
    nlp = spacy.blank("xx")
    doc_bin = DocBin()

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    words = []
    tags = []

    def create_doc(words, tags):
        text = " ".join(words)
        doc = nlp.make_doc(text)

        ents = []
        start = None
        label = None
        token_starts = [token.idx for token in doc]

        for i, tag in enumerate(tags):
            if tag.startswith("B-"):
                if start is not None:
                    span = doc.char_span(start, token_starts[i], label=label)
                    if span:
                        ents.append(span)

                start = token_starts[i]
                label = tag[2:]
            elif tag.startswith("I-"):
                continue
            else:
                if start is not None:
                    span = doc.char_span(start, token_starts[i], label=label)
                    if span:
                        ents.append(span)
                    start = None

        if start is not None:
            span = doc.char_span(start, len(text), label=label)
            if span:
                ents.append(span)

        doc.ents = ents
        return doc

    for line in lines:
        line = line.strip()

        if line == "":
            if words:
                doc_bin.add(create_doc(words, tags))
                words = []
                tags = []
        else:
            parts = line.split()
            if len(parts) < 2:
                continue
            words.append(parts[0])
            tags.append(parts[-1])

    if words:
        doc_bin.add(create_doc(words, tags))

    doc_bin.to_disk(output_file)


if __name__ == "__main__":
    convert_bio_to_spacy(
        "data/raw/adyan.txt",
        "data/processed/ner.spacy",
    )
