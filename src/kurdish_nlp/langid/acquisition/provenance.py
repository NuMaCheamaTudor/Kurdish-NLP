"""Record-level provenance sidecars linked to canonical dataset IDs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Any

from kurdish_nlp.langid.acquisition.schemas import (
    SOURCE_ID_PATTERN,
    VERSION_PATTERN,
    exact_fields,
    nonempty_string,
    optional_string,
    optional_timestamp,
    optional_url,
    sha256_or_none,
    strict_json_loads,
)
from kurdish_nlp.langid.dataset import DatasetRecord


def _json_parameters(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise TypeError("parameters must be a JSON object with string keys")
    try:
        encoded = json.dumps(value, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise ValueError("parameters must contain finite JSON values only") from error
    return json.loads(encoded)


@dataclass(frozen=True, slots=True)
class Transformation:
    name: str
    version: str
    timestamp: str | None = None
    parameters: dict[str, Any] | None = None
    input_reference: str | None = None
    output_reference: str | None = None

    def __post_init__(self) -> None:
        nonempty_string(self.name, "name")
        nonempty_string(self.version, "version")
        object.__setattr__(
            self, "timestamp", optional_timestamp(self.timestamp, "timestamp")
        )
        object.__setattr__(
            self,
            "parameters",
            _json_parameters(self.parameters if self.parameters is not None else {}),
        )
        object.__setattr__(
            self,
            "input_reference",
            optional_string(self.input_reference, "input_reference"),
        )
        object.__setattr__(
            self,
            "output_reference",
            optional_string(self.output_reference, "output_reference"),
        )

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> Transformation:
        exact_fields(
            value,
            {"name", "version"},
            {
                "timestamp",
                "parameters",
                "input_reference",
                "output_reference",
            },
        )
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "timestamp": self.timestamp,
            "parameters": self.parameters,
            "input_reference": self.input_reference,
            "output_reference": self.output_reference,
        }


@dataclass(frozen=True, slots=True)
class ProvenanceRecord:
    schema_version: int
    canonical_record_id: str
    source_id: str
    acquisition_manifest_version: str
    original_record_id: str | None = None
    original_document_id: str | None = None
    alignment_group_id: str | None = None
    original_url: str | None = None
    source_file: str | None = None
    source_row_number: int | None = None
    page_id: str | None = None
    revision_id: str | None = None
    author_id: str | None = None
    contributor_id: str | None = None
    translator_id: str | None = None
    language_variety: str | None = None
    county: str | None = None
    orthography: str | None = None
    original_language_label: str | None = None
    script: str | None = None
    original_split: str | None = None
    original_license: str | None = None
    original_text_sha256: str | None = None
    normalized_text_sha256: str | None = None
    source_snapshot: str | None = None
    transformations: tuple[Transformation, ...] = ()

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported provenance schema_version")
        nonempty_string(self.canonical_record_id, "canonical_record_id")
        if not SOURCE_ID_PATTERN.fullmatch(
            nonempty_string(self.source_id, "source_id")
        ):
            raise ValueError("invalid source_id")
        if not VERSION_PATTERN.fullmatch(
            nonempty_string(
                self.acquisition_manifest_version, "acquisition_manifest_version"
            )
        ):
            raise ValueError("invalid acquisition_manifest_version")
        for field_name in (
            "original_record_id",
            "original_document_id",
            "alignment_group_id",
            "source_file",
            "page_id",
            "revision_id",
            "author_id",
            "contributor_id",
            "translator_id",
            "language_variety",
            "county",
            "orthography",
            "original_language_label",
            "script",
            "original_split",
            "original_license",
            "source_snapshot",
        ):
            object.__setattr__(
                self, field_name, optional_string(getattr(self, field_name), field_name)
            )
        object.__setattr__(
            self, "original_url", optional_url(self.original_url, "original_url")
        )
        if self.source_row_number is not None and (
            type(self.source_row_number) is not int or self.source_row_number < 1
        ):
            raise ValueError("source_row_number must be an integer >= 1")
        object.__setattr__(
            self,
            "original_text_sha256",
            sha256_or_none(self.original_text_sha256, "original_text_sha256"),
        )
        object.__setattr__(
            self,
            "normalized_text_sha256",
            sha256_or_none(self.normalized_text_sha256, "normalized_text_sha256"),
        )
        if not isinstance(self.transformations, (tuple, list)) or any(
            not isinstance(item, Transformation) for item in self.transformations
        ):
            raise TypeError(
                "transformations must be an array of Transformation objects"
            )
        object.__setattr__(self, "transformations", tuple(self.transformations))

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> ProvenanceRecord:
        exact_fields(
            value,
            {
                "schema_version",
                "canonical_record_id",
                "source_id",
                "acquisition_manifest_version",
            },
            {
                "original_record_id",
                "original_document_id",
                "alignment_group_id",
                "original_url",
                "source_file",
                "source_row_number",
                "page_id",
                "revision_id",
                "author_id",
                "contributor_id",
                "translator_id",
                "language_variety",
                "county",
                "orthography",
                "original_language_label",
                "script",
                "original_split",
                "original_license",
                "original_text_sha256",
                "normalized_text_sha256",
                "source_snapshot",
                "transformations",
            },
        )
        return cls(
            **{
                **value,
                "transformations": tuple(
                    Transformation.from_mapping(item)
                    for item in value.get("transformations", [])
                ),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "canonical_record_id": self.canonical_record_id,
            "source_id": self.source_id,
            "acquisition_manifest_version": self.acquisition_manifest_version,
            "original_record_id": self.original_record_id,
            "original_document_id": self.original_document_id,
            "alignment_group_id": self.alignment_group_id,
            "original_url": self.original_url,
            "source_file": self.source_file,
            "source_row_number": self.source_row_number,
            "page_id": self.page_id,
            "revision_id": self.revision_id,
            "author_id": self.author_id,
            "contributor_id": self.contributor_id,
            "translator_id": self.translator_id,
            "language_variety": self.language_variety,
            "county": self.county,
            "orthography": self.orthography,
            "original_language_label": self.original_language_label,
            "script": self.script,
            "original_split": self.original_split,
            "original_license": self.original_license,
            "original_text_sha256": self.original_text_sha256,
            "normalized_text_sha256": self.normalized_text_sha256,
            "source_snapshot": self.source_snapshot,
            "transformations": [item.to_dict() for item in self.transformations],
        }


def validate_linkage(
    records: list[DatasetRecord], sidecars: list[ProvenanceRecord]
) -> None:
    """Require exactly one sidecar per canonical record ID."""
    canonical_ids = [record.id for record in records]
    sidecar_ids = [record.canonical_record_id for record in sidecars]
    if len(set(canonical_ids)) != len(canonical_ids):
        raise ValueError("canonical records contain duplicate IDs")
    if len(set(sidecar_ids)) != len(sidecar_ids):
        raise ValueError("sidecars contain duplicate canonical_record_id values")
    if set(canonical_ids) != set(sidecar_ids):
        raise ValueError("sidecar IDs must match canonical record IDs exactly")


def load_sidecars(path: str | Path) -> list[ProvenanceRecord]:
    records: list[ProvenanceRecord] = []
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                records.append(ProvenanceRecord.from_mapping(strict_json_loads(line)))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid sidecar line {line_number}: {error}"
                ) from error
    return records


def save_sidecars(records: list[ProvenanceRecord], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8", newline="\n") as output:
        output.writelines(
            json.dumps(
                record.to_dict(), ensure_ascii=False, sort_keys=True, allow_nan=False
            )
            + "\n"
            for record in records
        )


def validate_parallel_jsonl_linkage(
    canonical_path: str | Path, sidecar_path: str | Path
) -> int:
    """Validate aligned canonical/sidecar files without loading them into RAM."""
    count = 0
    with (
        Path(canonical_path).open(encoding="utf-8") as canonical,
        Path(sidecar_path).open(encoding="utf-8") as sidecars,
    ):
        for count, (canonical_line, sidecar_line) in enumerate(
            zip_longest(canonical, sidecars), 1
        ):
            if canonical_line is None or sidecar_line is None:
                raise ValueError(f"canonical/sidecar length mismatch at line {count}")
            record = DatasetRecord.from_mapping(strict_json_loads(canonical_line))
            sidecar = ProvenanceRecord.from_mapping(strict_json_loads(sidecar_line))
            if sidecar.canonical_record_id != record.id:
                raise ValueError(f"canonical/sidecar ID mismatch at line {count}")
            if (
                sidecar.alignment_group_id is not None
                and sidecar.alignment_group_id != record.document_id
            ):
                raise ValueError(f"canonical/sidecar group mismatch at line {count}")
            if (
                sidecar.normalized_text_sha256 is not None
                and sidecar.normalized_text_sha256
                != hashlib.sha256(record.text.encode("utf-8")).hexdigest()
            ):
                raise ValueError(
                    f"canonical/sidecar text hash mismatch at line {count}"
                )
            if (
                sidecar.source_snapshot is not None
                and record.source != f"{sidecar.source_id}:{sidecar.source_snapshot}"
            ):
                raise ValueError(f"canonical/sidecar source mismatch at line {count}")
    return count
