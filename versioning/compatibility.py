"""Compatibility vocabulary for caller-defined policies."""

from dataclasses import dataclass
from enum import StrEnum


class CompatibilityStatus(StrEnum):
    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class CompatibilityResult:
    status: CompatibilityStatus = CompatibilityStatus.UNKNOWN
    reason: str | None = None
