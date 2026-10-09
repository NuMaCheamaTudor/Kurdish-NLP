"""Offline source metadata, provenance and dataset-role policy checks."""

from kurdish_nlp.langid.acquisition.licenses import LicenseId, LicenseSpec
from kurdish_nlp.langid.acquisition.manifests import (
    SourceIndex,
    SourceManifest,
    load_indexed_manifests,
    load_manifest,
    save_manifest,
)
from kurdish_nlp.langid.acquisition.policy import PolicyDecision, evaluate
from kurdish_nlp.langid.acquisition.provenance import ProvenanceRecord, Transformation
from kurdish_nlp.langid.acquisition.schemas import (
    DatasetRole,
    PolicyProfile,
    ProvenanceStatus,
)

__all__ = [
    "DatasetRole",
    "LicenseId",
    "LicenseSpec",
    "PolicyDecision",
    "PolicyProfile",
    "ProvenanceRecord",
    "ProvenanceStatus",
    "SourceIndex",
    "SourceManifest",
    "Transformation",
    "evaluate",
    "load_indexed_manifests",
    "load_manifest",
    "save_manifest",
]
