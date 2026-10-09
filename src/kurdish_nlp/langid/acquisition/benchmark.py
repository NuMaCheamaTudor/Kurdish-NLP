"""Strict, evaluation-only record schema; deliberately incompatible with DatasetRecord."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kurdish_nlp.langid.acquisition.schemas import exact_fields, nonempty_string
from kurdish_nlp.langid.normalization import normalize_text

BENCHMARK = "ud-kurdish-v2.18"
PARTITION = "external_test"
FIELDS = {
    "id",
    "text",
    "expected_language",
    "benchmark",
    "treebank",
    "original_split",
    "evaluation_partition",
    "source",
    "license",
    "slice",
    "strict_external",
    "exclusion_reasons",
}


@dataclass(frozen=True, slots=True)
class ExternalBenchmarkRecord:
    id: str
    text: str
    expected_language: str
    benchmark: str
    treebank: str
    original_split: str
    evaluation_partition: str
    source: str
    license: str
    slice: str
    strict_external: bool
    exclusion_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in FIELDS - {"strict_external", "exclusion_reasons"}:
            nonempty_string(getattr(self, name), name)
        if self.expected_language not in {"sdh", "kmr"}:
            raise ValueError("unsupported UD benchmark language")
        if self.benchmark != BENCHMARK or self.evaluation_partition != PARTITION:
            raise ValueError(
                "UD records must remain in the external evaluation partition"
            )
        if self.original_split not in {"train", "dev", "test"}:
            raise ValueError("invalid original UD split")
        if self.source != "universal-dependencies" or self.license != "CC-BY-SA-4.0":
            raise ValueError("invalid UD source or license")
        if type(self.strict_external) is not bool:
            raise TypeError("strict_external must be boolean")
        if not isinstance(self.exclusion_reasons, (tuple, list)) or any(
            not isinstance(reason, str) or not reason
            for reason in self.exclusion_reasons
        ):
            raise TypeError("exclusion_reasons must be non-empty strings")
        object.__setattr__(self, "exclusion_reasons", tuple(self.exclusion_reasons))
        if self.strict_external == bool(self.exclusion_reasons):
            raise ValueError("strict_external and exclusion_reasons disagree")

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> ExternalBenchmarkRecord:
        exact_fields(value, FIELDS)
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return {
            name: list(getattr(self, name))
            if name == "exclusion_reasons"
            else getattr(self, name)
            for name in (
                "id",
                "text",
                "expected_language",
                "benchmark",
                "treebank",
                "original_split",
                "evaluation_partition",
                "source",
                "license",
                "slice",
                "strict_external",
                "exclusion_reasons",
            )
        }


def validate_benchmark(path: Path, provenance_path: Path) -> dict[str, int]:
    """Validate the two parallel files without admitting benchmark rows to training."""
    counts = {"records": 0, "strict_external": 0}
    ids: set[str] = set()
    with (
        path.open(encoding="utf-8") as records,
        provenance_path.open(encoding="utf-8") as sidecars,
    ):
        from itertools import zip_longest

        for number, (raw_record, raw_provenance) in enumerate(
            zip_longest(records, sidecars), 1
        ):
            if raw_record is None or raw_provenance is None:
                raise ValueError("benchmark/provenance row counts differ")
            record = ExternalBenchmarkRecord.from_mapping(json.loads(raw_record))
            provenance = json.loads(raw_provenance)
            if (
                not isinstance(provenance, dict)
                or provenance.get("evaluation_record_id") != record.id
            ):
                raise ValueError(f"provenance linkage failure at row {number}")
            if (
                provenance.get("treebank") != record.treebank
                or provenance.get("original_split") != record.original_split
            ):
                raise ValueError(f"provenance source mismatch at row {number}")
            if (
                provenance.get("license") != record.license
                or provenance.get("ud_release") != "v2.18"
            ):
                raise ValueError(f"provenance release/license mismatch at row {number}")
            if (
                provenance.get("benchmark_text_sha256")
                != hashlib.sha256(record.text.encode("utf-8")).hexdigest()
            ):
                raise ValueError(f"benchmark text hash mismatch at row {number}")
            if (
                provenance.get("normalized_text_sha256")
                != hashlib.sha256(
                    normalize_text(record.text).encode("utf-8")
                ).hexdigest()
            ):
                raise ValueError(f"normalized text hash mismatch at row {number}")
            if record.id in ids:
                raise ValueError(f"duplicate benchmark ID: {record.id}")
            ids.add(record.id)
            counts["records"] += 1
            counts["strict_external"] += record.strict_external
    if not ids:
        raise ValueError("empty benchmark")
    return counts
