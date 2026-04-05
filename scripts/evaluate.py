"""
Evaluate the trained Kurdish Kurmanji spaCy model.

Usage:
    python scripts/evaluate.py
"""

import spacy


def main():
    nlp = spacy.load("output/model-best")

    test_sentences = [
        "Ez diçim malê.",
        "Ew kitêbê dixwîne.",
    ]

    for text in test_sentences:
        doc = nlp(text)
        print(f"\nText: {text}")
        print(f"{'Token':<15} {'POS':<10} {'DEP':<15} {'Lemma':<15}")
        print("-" * 55)
        for token in doc:
            print(f"{token.text:<15} {token.pos_:<10} {token.dep_:<15} {token.lemma_:<15}")


if __name__ == "__main__":
    main()
