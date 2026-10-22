"""Download only the pinned XLM-R encoder files, then verify their SHA-256s.

Run from the repository root in the isolated NER environment. No datasets fetched.
"""

import json
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path("data/ner/cache/huggingface").resolve()))

from huggingface_hub import snapshot_download

from kurdish_nlp.ner.training import validate_encoder


def main():
    cache = Path("data/ner/cache")
    cache.mkdir(parents=True, exist_ok=True)
    spec = json.loads(Path("configs/ner/encoder-v2.json").read_text())
    path = Path(
        snapshot_download(
            spec["repo_id"],
            revision=spec["revision"],
            allow_patterns=list(spec["files"]),
            max_workers=2,
        )
    ).resolve()
    validate_encoder(path)
    (cache / "encoder-path.txt").write_text(str(path) + "\n")
    print(path)


if __name__ == "__main__":
    main()
