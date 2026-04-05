"""
Train the spaCy Kurdish Kurmanji pipeline.

Usage:
    python scripts/train.py
"""

import subprocess
import sys


def main():
    result = subprocess.run(
        [
            sys.executable, "-m", "spacy", "train",
            "configs/base.cfg",
            "--output", "output",
            "--paths.train", "data/processed/train.spacy",
            "--paths.dev", "data/processed/dev.spacy",
        ]
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
