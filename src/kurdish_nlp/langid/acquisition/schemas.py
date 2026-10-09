"""Shared, dependency-free types and validation for acquisition metadata."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from typing import Any
from urllib.parse import urlsplit

SCHEMA_VERSION = 1
SOURCE_ID_PATTERN = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
VERSION_PATTERN = re.compile(r"[1-9][0-9]*(?:\.[0-9]+)*\Z")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


class DatasetRole(StrEnum):
    TRAINING = "training"
    DEVELOPMENT = "development"
    CALIBRATION = "calibration"
    TEST = "test"
    EXTERNAL_EVALUATION = "external_evaluation"
    BENCHMARK = "benchmark"
    PROVENANCE_ONLY = "provenance_only"


class ProvenanceStatus(StrEnum):
    VERIFIED = "verified"
    VERIFIED_WITH_CAVEATS = "verified_with_caveats"
    UNVERIFIED = "unverified"
    PROHIBITED = "prohibited"


class PolicyProfile(StrEnum):
    COMMERCIAL = "commercial"
    RESEARCH = "research"


class Permission(StrEnum):
    ALLOWED = "allowed"
    PROHIBITED = "prohibited"
    UNKNOWN = "unknown"


class ConditionalPermission(StrEnum):
    ALLOWED = "allowed"
    PROHIBITED = "prohibited"
    CONDITIONAL = "conditional"
    UNKNOWN = "unknown"


def exact_fields(
    value: Mapping[str, Any], required: set[str], optional: set[str] = frozenset()
) -> None:
    if not isinstance(value, Mapping):
        raise TypeError("expected a JSON object")
    missing = required - value.keys()
    unknown = value.keys() - required - optional
    if missing:
        raise ValueError(f"missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"unknown fields: {', '.join(sorted(unknown))}")


def nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field} must be a non-empty, trimmed string")
    return value


def optional_string(value: Any, field: str) -> str | None:
    return None if value is None else nonempty_string(value, field)


def valid_url(value: Any, field: str) -> str:
    url = nonempty_string(value, field)
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or any(character.isspace() for character in url)
    ):
        raise ValueError(
            f"{field} must be an HTTP(S) URL with a host and no credentials"
        )
    try:
        _ = parsed.port
    except ValueError as error:
        raise ValueError(f"{field} has an invalid port") from error
    return url


def optional_url(value: Any, field: str) -> str | None:
    return None if value is None else valid_url(value, field)


def aware_timestamp(value: Any, field: str) -> str:
    timestamp = nonempty_string(value, field)
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError as error:
        raise ValueError(f"{field} must be an ISO 8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone offset")
    return timestamp


def optional_timestamp(value: Any, field: str) -> str | None:
    return None if value is None else aware_timestamp(value, field)


def sha256_or_none(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{field} must be 64 lowercase hexadecimal characters")
    return value


def string_tuple(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{field} must be an array")
    return tuple(nonempty_string(item, field) for item in value)


def strict_json_loads(raw: str) -> Any:
    """Reject duplicate object keys and non-standard numeric constants."""

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"invalid JSON constant: {value}")

    return json.loads(
        raw, object_pairs_hook=unique_pairs, parse_constant=reject_constant
    )
