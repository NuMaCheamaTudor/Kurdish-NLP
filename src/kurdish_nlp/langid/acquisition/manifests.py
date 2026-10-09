"""Versioned source manifests and a reviewable index of approved source files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kurdish_nlp.langid.acquisition.licenses import LicenseSpec
from kurdish_nlp.langid.acquisition.schemas import (
    SCHEMA_VERSION,
    SOURCE_ID_PATTERN,
    VERSION_PATTERN,
    DatasetRole,
    ProvenanceStatus,
    exact_fields,
    nonempty_string,
    optional_string,
    optional_timestamp,
    sha256_or_none,
    strict_json_loads,
    string_tuple,
    valid_url,
)


def _schema_version(value: Any) -> int:
    if type(value) is not int or value != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version: {value!r}; expected {SCHEMA_VERSION}"
        )
    return value


@dataclass(frozen=True, slots=True)
class SourceManifest:
    schema_version: int
    manifest_version: str
    source_id: str
    source_name: str
    source_url: str
    source_version: str
    retrieved_at: str | None
    checksum_sha256: str | None
    license: LicenseSpec
    provenance_status: ProvenanceStatus
    allowed_roles: tuple[DatasetRole, ...]
    sidecar_reference: str | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _schema_version(self.schema_version)
        if not VERSION_PATTERN.fullmatch(
            nonempty_string(self.manifest_version, "manifest_version")
        ):
            raise ValueError("manifest_version must be a numeric dotted version")
        if not SOURCE_ID_PATTERN.fullmatch(
            nonempty_string(self.source_id, "source_id")
        ):
            raise ValueError("source_id must be stable lowercase kebab-case")
        nonempty_string(self.source_name, "source_name")
        valid_url(self.source_url, "source_url")
        nonempty_string(self.source_version, "source_version")
        object.__setattr__(
            self, "retrieved_at", optional_timestamp(self.retrieved_at, "retrieved_at")
        )
        object.__setattr__(
            self,
            "checksum_sha256",
            sha256_or_none(self.checksum_sha256, "checksum_sha256"),
        )
        if (self.retrieved_at is None) != (self.checksum_sha256 is None):
            raise ValueError(
                "retrieved_at and checksum_sha256 must both be present for acquired data"
            )
        if not isinstance(self.license, LicenseSpec):
            raise TypeError("license must be a LicenseSpec")
        try:
            object.__setattr__(
                self, "provenance_status", ProvenanceStatus(self.provenance_status)
            )
        except ValueError as error:
            raise ValueError("invalid provenance_status") from error
        if not isinstance(self.allowed_roles, (tuple, list)):
            raise TypeError("allowed_roles must be an array")
        try:
            roles = tuple(DatasetRole(role) for role in self.allowed_roles)
        except ValueError as error:
            raise ValueError("invalid allowed_roles") from error
        if len(set(roles)) != len(roles):
            raise ValueError("allowed_roles must not contain duplicates")
        object.__setattr__(self, "allowed_roles", roles)
        object.__setattr__(
            self,
            "sidecar_reference",
            optional_string(self.sidecar_reference, "sidecar_reference"),
        )
        object.__setattr__(self, "notes", string_tuple(self.notes, "notes"))
        if self.provenance_status is ProvenanceStatus.PROHIBITED and any(
            role is not DatasetRole.PROVENANCE_ONLY for role in roles
        ):
            raise ValueError("prohibited provenance may only allow provenance_only")

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> SourceManifest:
        exact_fields(
            value,
            {
                "schema_version",
                "manifest_version",
                "source_id",
                "source_name",
                "source_url",
                "source_version",
                "retrieved_at",
                "checksum_sha256",
                "license",
                "provenance_status",
                "allowed_roles",
                "sidecar_reference",
                "notes",
            },
        )
        return cls(
            schema_version=value["schema_version"],
            manifest_version=value["manifest_version"],
            source_id=value["source_id"],
            source_name=value["source_name"],
            source_url=value["source_url"],
            source_version=value["source_version"],
            retrieved_at=value["retrieved_at"],
            checksum_sha256=value["checksum_sha256"],
            license=LicenseSpec.from_mapping(value["license"]),
            provenance_status=value["provenance_status"],
            allowed_roles=value["allowed_roles"],
            sidecar_reference=value["sidecar_reference"],
            notes=value["notes"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "manifest_version": self.manifest_version,
            "source_id": self.source_id,
            "source_name": self.source_name,
            "source_url": self.source_url,
            "source_version": self.source_version,
            "retrieved_at": self.retrieved_at,
            "checksum_sha256": self.checksum_sha256,
            "license": self.license.to_dict(),
            "provenance_status": self.provenance_status.value,
            "allowed_roles": [role.value for role in self.allowed_roles],
            "sidecar_reference": self.sidecar_reference,
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class SourceIndex:
    schema_version: int
    manifests: tuple[str, ...]

    def __post_init__(self) -> None:
        _schema_version(self.schema_version)
        paths = string_tuple(self.manifests, "manifests")
        if len(set(paths)) != len(paths):
            raise ValueError("manifest index contains duplicate references")
        for path in paths:
            candidate = Path(path)
            if (
                candidate.is_absolute()
                or ".." in candidate.parts
                or candidate.suffix != ".json"
            ):
                raise ValueError(
                    "manifest references must be relative .json paths without '..'"
                )
        object.__setattr__(self, "manifests", paths)

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> SourceIndex:
        exact_fields(value, {"schema_version", "manifests"})
        return cls(value["schema_version"], value["manifests"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "manifests": list(self.manifests),
        }


def load_manifest(path: str | Path) -> SourceManifest:
    with Path(path).open("r", encoding="utf-8") as source:
        value = strict_json_loads(source.read())
    return SourceManifest.from_mapping(value)


def save_manifest(manifest: SourceManifest, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8", newline="\n") as output:
        json.dump(manifest.to_dict(), output, ensure_ascii=False, indent=2)
        output.write("\n")


def load_index(path: str | Path) -> SourceIndex:
    with Path(path).open("r", encoding="utf-8") as source:
        return SourceIndex.from_mapping(strict_json_loads(source.read()))


def load_indexed_manifests(path: str | Path) -> tuple[SourceManifest, ...]:
    """Load an index's reviewed relative references without network access."""
    index_path = Path(path)
    base = index_path.parent.resolve()
    manifests: list[SourceManifest] = []
    for reference in load_index(index_path).manifests:
        target = (base / reference).resolve()
        if not target.is_relative_to(base):
            raise ValueError("manifest reference escapes index directory")
        manifests.append(load_manifest(target))
    source_ids = [manifest.source_id for manifest in manifests]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("manifest index contains duplicate source_id values")
    return tuple(manifests)


def save_index(index: SourceIndex, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8", newline="\n") as output:
        json.dump(index.to_dict(), output, ensure_ascii=False, indent=2)
        output.write("\n")
