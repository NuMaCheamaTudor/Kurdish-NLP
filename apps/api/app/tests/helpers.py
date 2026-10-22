from __future__ import annotations

from dataclasses import replace

import pytest

from app.config import DEFAULT_ARTIFACTS_DIR, Settings

# One Dataset V1 *train* sentence per class (Tatoeba, CC BY 2.0 FR). They check
# the end-to-end label mapping, not accuracy: the models saw them in training.
# Held-out test/dev and UD benchmark sentences are deliberately not used.
TRAIN_SAMPLES = {
    "ckb": "زۆر جار ناوی ئەو کەسانەم لەبیر دەچێتەوە کە چاوم پێیان دەکەوێت.",
    "kmr": "Koma dîlanê govendên gelêrî yên cur bi cur gerandin.",
    "sdh": "مە تووام بچم ئەڕا ئەورا و بامەو ئەڕا ئێرا.",
    "ar": "كان ساني يحبّ أن يأتي إلى المنزل كي يرى أبناءه.",
    "fa": "تام و مری روی میزی در گوشه‌ای خلوت هستند.",
    "tr": "Bu saçma konuyu sonuna kadar dinlediğin için teşekkür ederim.",
    "en": "Tom is no longer the man he used to be.",
}

ARTIFACTS_PRESENT = (DEFAULT_ARTIFACTS_DIR / "tfidf_lr" / "model.joblib").is_file() and (
    DEFAULT_ARTIFACTS_DIR / "fasttext" / "model.bin"
).is_file()

requires_artifacts = pytest.mark.skipif(
    not ARTIFACTS_PRESENT, reason="baseline-v1 model artifacts are not present"
)


def make_settings(**overrides: object) -> Settings:
    base = Settings(ner_model_path=None, grammar_model_path=None)
    return replace(base, **overrides)  # type: ignore[arg-type]
