"""Canonical JSONL dataset validation, analysis, and conversion utilities."""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kurdish_nlp.langid.normalization import normalize_text
from kurdish_nlp.langid.schemas import TRAINABLE_LANGUAGE_CODES, LanguageCode

CANONICAL_FIELDS = frozenset(
    {
        "id",
        "text",
        "label",
        "source",
        "domain",
        "document_id",
        "license",
        "split",
    }
)
REQUIRED_FIELDS = CANONICAL_FIELDS
VALID_SPLITS = frozenset({"train", "dev", "test"})


@dataclass(frozen=True, slots=True)
class DatasetRecord:
    id: str
    text: str
    label: LanguageCode
    source: str
    domain: str
    document_id: str
    license: str
    split: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> DatasetRecord:
        missing = sorted(REQUIRED_FIELDS - value.keys())
        unknown = sorted(value.keys() - CANONICAL_FIELDS)
        if missing:
            raise ValueError(f"missing required fields: {', '.join(missing)}")
        if unknown:
            raise ValueError(f"unknown fields: {', '.join(unknown)}")

        strings: dict[str, str] = {}
        for field_name in CANONICAL_FIELDS - {"label"}:
            field_value = value[field_name]
            if not isinstance(field_value, str):
                raise TypeError(f"{field_name} must be a string")
            strings[field_name] = field_value.strip()

        for field_name in ("id", "text", "source", "domain", "document_id", "license"):
            if not strings[field_name]:
                raise ValueError(f"{field_name} must be non-empty")

        try:
            label = LanguageCode(value["label"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"unsupported label: {value['label']!r}") from error
        if label not in TRAINABLE_LANGUAGE_CODES:
            raise ValueError(
                "und is a runtime abstention result and is not a trainable label"
            )
        if strings["split"] not in VALID_SPLITS:
            raise ValueError(f"split must be one of {', '.join(sorted(VALID_SPLITS))}")

        return cls(label=label, **strings)

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "text": self.text,
            "label": self.label.value,
            "source": self.source,
            "domain": self.domain,
            "document_id": self.document_id,
            "license": self.license,
            "split": self.split,
        }


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: str
    message: str
    line_number: int | None = None
    record_ids: tuple[str, ...] = ()

    def __str__(self) -> str:
        location = f"line {self.line_number}: " if self.line_number else ""
        records = f" [{', '.join(self.record_ids)}]" if self.record_ids else ""
        return f"{location}{self.code}: {self.message}{records}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "line_number": self.line_number,
            "record_ids": list(self.record_ids),
        }


class DatasetValidationError(ValueError):
    """Raised with every detected issue rather than discarding bad records."""

    def __init__(self, issues: Sequence[ValidationIssue]) -> None:
        self.issues = tuple(issues)
        super().__init__("dataset validation failed:\n" + "\n".join(map(str, issues)))


@dataclass(frozen=True, slots=True)
class DatasetAnalysis:
    record_count: int
    label_distribution: dict[str, int]
    split_distribution: dict[str, int]
    character_lengths: dict[str, float]
    token_lengths: dict[str, float]
    duplicate_texts: tuple[tuple[str, ...], ...]
    text_split_leakage: tuple[tuple[str, ...], ...]
    document_split_leakage: tuple[tuple[str, ...], ...]
    issues: tuple[ValidationIssue, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_count": self.record_count,
            "label_distribution": self.label_distribution,
            "split_distribution": self.split_distribution,
            "character_lengths": self.character_lengths,
            "token_lengths": self.token_lengths,
            "duplicate_text_record_ids": [
                list(group) for group in self.duplicate_texts
            ],
            "text_split_leakage_record_ids": [
                list(group) for group in self.text_split_leakage
            ],
            "document_split_leakage_record_ids": [
                list(group) for group in self.document_split_leakage
            ],
            "issues": [issue.to_dict() for issue in self.issues],
        }


def load_jsonl(path: str | Path) -> list[DatasetRecord]:
    """Load canonical records and raise once with all line-level schema errors."""

    source_path = Path(path)
    records: list[DatasetRecord] = []
    issues: list[ValidationIssue] = []

    with source_path.open("r", encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                issues.append(
                    ValidationIssue(
                        code="blank_line",
                        message="blank lines are not valid JSONL records",
                        line_number=line_number,
                    )
                )
                continue
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as error:
                issues.append(
                    ValidationIssue(
                        code="invalid_json",
                        message=error.msg,
                        line_number=line_number,
                    )
                )
                continue
            if not isinstance(value, dict):
                issues.append(
                    ValidationIssue(
                        code="invalid_record",
                        message="each JSONL value must be an object",
                        line_number=line_number,
                    )
                )
                continue
            try:
                records.append(DatasetRecord.from_mapping(value))
            except (TypeError, ValueError) as error:
                issues.append(
                    ValidationIssue(
                        code="invalid_record",
                        message=str(error),
                        line_number=line_number,
                        record_ids=(str(value.get("id", "<missing>")),),
                    )
                )

    if issues:
        raise DatasetValidationError(issues)
    if not records:
        raise DatasetValidationError(
            [ValidationIssue(code="empty_dataset", message="dataset has no records")]
        )
    return records


def find_exact_duplicate_texts(
    records: Iterable[DatasetRecord],
) -> tuple[tuple[str, ...], ...]:
    """Find texts equal after conservative normalization."""

    by_text: dict[str, list[str]] = defaultdict(list)
    for record in records:
        by_text[normalize_text(record.text)].append(record.id)
    return tuple(
        tuple(record_ids) for record_ids in by_text.values() if len(record_ids) > 1
    )


def find_text_split_leakage(
    records: Iterable[DatasetRecord],
) -> tuple[tuple[str, ...], ...]:
    by_text: dict[str, list[DatasetRecord]] = defaultdict(list)
    for record in records:
        by_text[normalize_text(record.text)].append(record)
    return tuple(
        tuple(item.id for item in group)
        for group in by_text.values()
        if len({item.split for item in group}) > 1
    )


def find_document_split_leakage(
    records: Iterable[DatasetRecord],
) -> tuple[tuple[str, ...], ...]:
    by_document: dict[str, list[DatasetRecord]] = defaultdict(list)
    for record in records:
        by_document[record.document_id].append(record)
    return tuple(
        tuple(item.id for item in group)
        for group in by_document.values()
        if len({item.split for item in group}) > 1
    )


def analyze_records(records: Sequence[DatasetRecord]) -> DatasetAnalysis:
    """Summarize a dataset and report duplicate/leakage validation issues."""

    issues: list[ValidationIssue] = []
    ids: dict[str, list[str]] = defaultdict(list)
    for record in records:
        ids[record.id].append(record.id)
    for record_id, occurrences in ids.items():
        if len(occurrences) > 1:
            issues.append(
                ValidationIssue(
                    code="duplicate_id",
                    message="record ID occurs more than once",
                    record_ids=(record_id,),
                )
            )

    duplicate_texts = find_exact_duplicate_texts(records)
    text_leakage = find_text_split_leakage(records)
    document_leakage = find_document_split_leakage(records)
    issues.extend(
        ValidationIssue(
            code="duplicate_text",
            message="normalized text occurs more than once",
            record_ids=group,
        )
        for group in duplicate_texts
    )
    issues.extend(
        ValidationIssue(
            code="text_split_leakage",
            message="the same normalized text occurs in multiple splits",
            record_ids=group,
        )
        for group in text_leakage
    )
    issues.extend(
        ValidationIssue(
            code="document_split_leakage",
            message="document_id occurs in multiple splits",
            record_ids=group,
        )
        for group in document_leakage
    )

    character_lengths = [len(normalize_text(record.text)) for record in records]
    token_lengths = [len(normalize_text(record.text).split()) for record in records]
    return DatasetAnalysis(
        record_count=len(records),
        label_distribution=dict(
            sorted(Counter(record.label.value for record in records).items())
        ),
        split_distribution=dict(
            sorted(Counter(record.split for record in records).items())
        ),
        character_lengths=_length_summary(character_lengths),
        token_lengths=_length_summary(token_lengths),
        duplicate_texts=duplicate_texts,
        text_split_leakage=text_leakage,
        document_split_leakage=document_leakage,
        issues=tuple(issues),
    )


def validate_jsonl(path: str | Path) -> DatasetAnalysis:
    records = load_jsonl(path)
    analysis = analyze_records(records)
    if analysis.issues:
        raise DatasetValidationError(analysis.issues)
    return analysis


def convert_jsonl_to_fasttext(
    input_path: str | Path,
    output_path: str | Path,
    *,
    splits: Iterable[str] | None = None,
) -> int:
    """Validate canonical JSONL and write one fastText example per record."""

    records = load_jsonl(input_path)
    analysis = analyze_records(records)
    if analysis.issues:
        raise DatasetValidationError(analysis.issues)

    selected_splits = set(splits or VALID_SPLITS)
    invalid_splits = selected_splits - VALID_SPLITS
    if invalid_splits:
        raise ValueError(f"unsupported splits: {', '.join(sorted(invalid_splits))}")

    selected = [record for record in records if record.split in selected_splits]
    if not selected:
        raise ValueError("no records match the requested splits")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as output:
        for record in selected:
            output.write(
                f"__label__{record.label.value} {normalize_text(record.text)}\n"
            )
    return len(selected)


def _length_summary(values: Sequence[int]) -> dict[str, float]:
    return {
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": statistics.fmean(values),
        "median": float(statistics.median(values)),
    }
