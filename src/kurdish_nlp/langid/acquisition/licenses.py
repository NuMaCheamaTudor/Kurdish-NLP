"""Explicit engineering interpretation of a small set of source licenses.

This registry is a policy aid, not a legal determination of upstream ownership.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from kurdish_nlp.langid.acquisition.schemas import (
    ConditionalPermission,
    Permission,
    exact_fields,
    optional_url,
    string_tuple,
)


class LicenseId(StrEnum):
    MIT = "MIT"
    CC0 = "CC0-1.0"
    CC_BY_2 = "CC-BY-2.0"
    CC_BY_2_FR = "CC-BY-2.0-FR"
    CC_BY_4 = "CC-BY-4.0"
    CC_BY_SA_4 = "CC-BY-SA-4.0"
    CC_BY_ND_4 = "CC-BY-ND-4.0"
    CC_BY_NC_SA_4 = "CC-BY-NC-SA-4.0"
    CC_BY_NC_ND_4 = "CC-BY-NC-ND-4.0"
    UNKNOWN = "unknown"


# commercial, redistribution, derivatives, attribution, share-alike
LICENSE_RIGHTS: dict[
    LicenseId,
    tuple[Permission, ConditionalPermission, ConditionalPermission, bool, bool],
] = {
    LicenseId.MIT: (
        Permission.ALLOWED,
        ConditionalPermission.CONDITIONAL,
        ConditionalPermission.ALLOWED,
        True,
        False,
    ),
    LicenseId.CC0: (
        Permission.ALLOWED,
        ConditionalPermission.ALLOWED,
        ConditionalPermission.ALLOWED,
        False,
        False,
    ),
    LicenseId.CC_BY_2: (
        Permission.ALLOWED,
        ConditionalPermission.CONDITIONAL,
        ConditionalPermission.ALLOWED,
        True,
        False,
    ),
    LicenseId.CC_BY_2_FR: (
        Permission.ALLOWED,
        ConditionalPermission.CONDITIONAL,
        ConditionalPermission.ALLOWED,
        True,
        False,
    ),
    LicenseId.CC_BY_4: (
        Permission.ALLOWED,
        ConditionalPermission.CONDITIONAL,
        ConditionalPermission.ALLOWED,
        True,
        False,
    ),
    LicenseId.CC_BY_SA_4: (
        Permission.ALLOWED,
        ConditionalPermission.CONDITIONAL,
        ConditionalPermission.CONDITIONAL,
        True,
        True,
    ),
    LicenseId.CC_BY_ND_4: (
        Permission.ALLOWED,
        ConditionalPermission.CONDITIONAL,
        ConditionalPermission.PROHIBITED,
        True,
        False,
    ),
    LicenseId.CC_BY_NC_SA_4: (
        Permission.PROHIBITED,
        ConditionalPermission.CONDITIONAL,
        ConditionalPermission.CONDITIONAL,
        True,
        True,
    ),
    LicenseId.CC_BY_NC_ND_4: (
        Permission.PROHIBITED,
        ConditionalPermission.CONDITIONAL,
        ConditionalPermission.PROHIBITED,
        True,
        False,
    ),
    LicenseId.UNKNOWN: (
        Permission.UNKNOWN,
        ConditionalPermission.UNKNOWN,
        ConditionalPermission.UNKNOWN,
        False,
        False,
    ),
}


@dataclass(frozen=True, slots=True)
class LicenseSpec:
    identifier: LicenseId
    commercial_use: Permission
    redistribution: ConditionalPermission
    derivatives: ConditionalPermission
    attribution_required: bool
    share_alike_required: bool
    verification_url: str | None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, enum_type in (
            ("identifier", LicenseId),
            ("commercial_use", Permission),
            ("redistribution", ConditionalPermission),
            ("derivatives", ConditionalPermission),
        ):
            value = getattr(self, name)
            try:
                object.__setattr__(self, name, enum_type(value))
            except ValueError as error:
                raise ValueError(f"invalid {name}: {value!r}") from error
        for name in ("attribution_required", "share_alike_required"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        object.__setattr__(
            self,
            "verification_url",
            optional_url(self.verification_url, "verification_url"),
        )
        object.__setattr__(self, "notes", string_tuple(self.notes, "notes"))
        actual = (
            self.commercial_use,
            self.redistribution,
            self.derivatives,
            self.attribution_required,
            self.share_alike_required,
        )
        if actual != LICENSE_RIGHTS[self.identifier]:
            raise ValueError(
                f"rights contradict the {self.identifier.value} license preset"
            )
        if self.identifier is not LicenseId.UNKNOWN and self.verification_url is None:
            raise ValueError("known licenses require a verification_url")
        if self.identifier is LicenseId.UNKNOWN and self.verification_url is not None:
            raise ValueError("unknown license cannot have a license verification URL")

    @classmethod
    def for_identifier(
        cls,
        identifier: LicenseId | str,
        verification_url: str | None,
        notes: tuple[str, ...] = (),
    ) -> LicenseSpec:
        license_id = LicenseId(identifier)
        return cls(license_id, *LICENSE_RIGHTS[license_id], verification_url, notes)

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> LicenseSpec:
        exact_fields(
            value,
            {
                "identifier",
                "commercial_use",
                "redistribution",
                "derivatives",
                "attribution_required",
                "share_alike_required",
                "verification_url",
                "notes",
            },
        )
        return cls(
            identifier=value["identifier"],
            commercial_use=value["commercial_use"],
            redistribution=value["redistribution"],
            derivatives=value["derivatives"],
            attribution_required=value["attribution_required"],
            share_alike_required=value["share_alike_required"],
            verification_url=value["verification_url"],
            notes=value["notes"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier.value,
            "commercial_use": self.commercial_use.value,
            "redistribution": self.redistribution.value,
            "derivatives": self.derivatives.value,
            "attribution_required": self.attribution_required,
            "share_alike_required": self.share_alike_required,
            "verification_url": self.verification_url,
            "notes": list(self.notes),
        }
