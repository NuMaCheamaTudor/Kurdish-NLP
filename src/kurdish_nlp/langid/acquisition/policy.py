"""Central project policy for use of a source in a dataset role."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kurdish_nlp.langid.acquisition.manifests import SourceManifest
from kurdish_nlp.langid.acquisition.schemas import (
    ConditionalPermission,
    DatasetRole,
    Permission,
    PolicyProfile,
    ProvenanceStatus,
)

DERIVED_ARTIFACT_ROLES = frozenset(
    {
        DatasetRole.TRAINING,
        DatasetRole.DEVELOPMENT,
        DatasetRole.CALIBRATION,
    }
)


@dataclass(frozen=True, slots=True)
class PolicyFinding:
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    allowed: bool
    source_id: str
    role: DatasetRole
    profile: PolicyProfile
    reasons: tuple[PolicyFinding, ...]
    warnings: tuple[PolicyFinding, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "source_id": self.source_id,
            "role": self.role.value,
            "profile": self.profile.value,
            "reasons": [finding.to_dict() for finding in self.reasons],
            "warnings": [finding.to_dict() for finding in self.warnings],
        }


def evaluate(
    manifest: SourceManifest,
    *,
    role: DatasetRole | str,
    profile: PolicyProfile | str = PolicyProfile.COMMERCIAL,
) -> PolicyDecision:
    """Evaluate project use, independently of the upstream license declaration."""

    selected_role = DatasetRole(role)
    selected_profile = PolicyProfile(profile)
    reasons: list[PolicyFinding] = []
    warnings: list[PolicyFinding] = []

    def block(code: str, message: str) -> None:
        reasons.append(PolicyFinding(code, message))

    def warn(code: str, message: str) -> None:
        warnings.append(PolicyFinding(code, message))

    if selected_role not in manifest.allowed_roles:
        block("role_not_allowed", "project manifest does not allow this dataset role")

    # Recording bibliographic provenance does not consume the source data.
    if selected_role is DatasetRole.PROVENANCE_ONLY:
        return PolicyDecision(
            not reasons,
            manifest.source_id,
            selected_role,
            selected_profile,
            tuple(reasons),
            tuple(warnings),
        )

    if manifest.provenance_status is ProvenanceStatus.PROHIBITED:
        block("provenance_prohibited", "source provenance is prohibited")
    elif manifest.provenance_status is ProvenanceStatus.UNVERIFIED:
        block("provenance_unverified", "source provenance is unverified")
    elif manifest.provenance_status is ProvenanceStatus.VERIFIED_WITH_CAVEATS:
        warn("provenance_caveats", "review the source provenance caveats")

    license_spec = manifest.license
    if (
        license_spec.identifier.value == "unknown"
        or license_spec.verification_url is None
    ):
        block("license_unverified", "source license has not been verified")

    if selected_profile is PolicyProfile.COMMERCIAL:
        if license_spec.commercial_use is not Permission.ALLOWED:
            block(
                "commercial_use_not_allowed", "commercial use is prohibited or unknown"
            )
    elif license_spec.commercial_use is Permission.PROHIBITED:
        warn(
            "research_only",
            "license prohibits commercial use; keep this build research-only",
        )

    if selected_role in DERIVED_ARTIFACT_ROLES:
        if license_spec.derivatives is ConditionalPermission.PROHIBITED:
            block(
                "derivatives_prohibited",
                "this project policy requires derivatives for this role",
            )
        elif license_spec.derivatives is ConditionalPermission.UNKNOWN:
            block("derivatives_unknown", "derivative permission is unknown")
        elif license_spec.derivatives is ConditionalPermission.CONDITIONAL:
            warn("derivatives_conditional", "review derivative-work conditions")

    if license_spec.redistribution is ConditionalPermission.CONDITIONAL:
        warn(
            "redistribution_conditional",
            "review conditions before redistributing source data",
        )
    elif license_spec.redistribution is ConditionalPermission.PROHIBITED:
        warn("redistribution_prohibited", "do not redistribute the source data")
    elif license_spec.redistribution is ConditionalPermission.UNKNOWN:
        warn("redistribution_unknown", "redistribution permission is unknown")
    if license_spec.attribution_required:
        warn("attribution_required", "retain required attribution and license notice")
    if license_spec.share_alike_required:
        warn(
            "share_alike_required", "review share-alike obligations before distribution"
        )

    return PolicyDecision(
        not reasons,
        manifest.source_id,
        selected_role,
        selected_profile,
        tuple(reasons),
        tuple(warnings),
    )
