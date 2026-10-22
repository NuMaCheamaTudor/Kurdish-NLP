"""Abstention formatting with a fixed-probability classifier (no artifacts needed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from kurdish_nlp.langid.experiments.detector import CalibratedLanguageDetector
from kurdish_nlp.langid.reliability import ReliabilityPolicy

from app.services.langid import REASON_MESSAGES, LanguageIdService, LoadedModel

LABELS = ("ckb", "kmr", "sdh", "ar", "fa", "tr", "en")


class FixedClassifier:
    labels = LABELS

    def __init__(self, row: list[float]) -> None:
        self.row = np.asarray(row, dtype=np.float64)

    def predict_proba(self, texts):
        return np.tile(self.row, (len(texts), 1))


def service_with(row: list[float], tmp_path: Path) -> tuple[LanguageIdService, LoadedModel]:
    service = LanguageIdService(tmp_path, "tfidf_lr")
    detector = CalibratedLanguageDetector(
        FixedClassifier(row),
        temperature=1.0,
        policy=ReliabilityPolicy(confidence_threshold=0.4, margin_threshold=0.2),
        probability_floor=1e-12,
        model_identifier="fixed",
        model_version="0",
    )
    model = service.models["tfidf_lr"]
    model.tuned = model.reference = detector
    return service, model


def test_low_margin_abstention(tmp_path: Path) -> None:
    service, model = service_with([0.48, 0.0, 0.42, 0.1, 0.0, 0.0, 0.0], tmp_path)
    body = service.detect("کوردی کوردی", model, "tuned", 7)
    assert body.language.code == "und" and body.language.name == "Inconclusive"
    assert body.abstained and not body.is_reliable
    assert body.reason == "low_margin"
    assert body.reason_message == REASON_MESSAGES["low_margin"]
    assert body.confidence == 0.48
    assert [s.code for s in body.scores[:2]] == ["ckb", "sdh"]
    assert body.margin == np.float64(0.48) - np.float64(0.42)


def test_reliable_result(tmp_path: Path) -> None:
    service, model = service_with([0.05, 0.9, 0.05, 0, 0, 0, 0], tmp_path)
    body = service.detect("Ez diçim malê û tu jî werî", model, "tuned", 3)
    assert body.language.code == "kmr" and body.is_reliable and body.reason is None
    assert len(body.scores) == 3
