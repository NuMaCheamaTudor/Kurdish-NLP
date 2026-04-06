import spacy
from spacy.training import Corpus

nlp = spacy.blank("xx")

corpus = Corpus("data/processed/train.spacy")

examples = list(corpus(nlp))

print(f"Number of examples: {len(examples)}")

if examples:
    example = examples[0]
    print(f"Gold doc length: {len(example.y)}")
    for token in example.y[:5]:
        print(f"  {token.text}: pos={token.pos}, pos_={token.pos_}")