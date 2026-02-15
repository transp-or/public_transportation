from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class Severity(str, Enum):
    """Severity level for validation issues."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(slots=True, frozen=True)
class Issue:
    """
    A structured validation issue suitable for GUI display.

    :param severity: Severity level (error/warning/info).
    :param code: Stable machine-readable code, e.g. "STOP_LAT_RANGE".
    :param message: Human-readable explanation.
    :param location: A string pointing to where the issue occurred (e.g. "stops[STOP_123].lat").
    :param suggestion: Optional suggestion on how to fix the issue.
    :param context: Optional extra context (e.g. offending value), JSON-serializable if possible.
    """
    severity: Severity
    code: str
    message: str
    location: str
    suggestion: str | None = None
    context: dict[str, Any] | None = None


@dataclass(slots=True)
class ValidationReport:
    """
    Aggregates validation issues.

    :param issues: List of issues found during validation.
    """
    issues: list[Issue]

    @property
    def ok(self) -> bool:
        """True if no ERROR issues are present."""
        return all(i.severity != Severity.ERROR for i in self.issues)

    def extend(self, other: ValidationReport) -> None:
        """Append issues from another report."""
        self.issues.extend(other.issues)

    def add(self, issue: Issue) -> None:
        """Add one issue."""
        self.issues.append(issue)

    def to_text(self, *, max_issues: int | None = 200) -> str:
        """
        Render a human-readable summary.

        :param max_issues: Maximum number of issues to show.
        :return: Multi-line string.
        """
        issues = self.issues if max_issues is None else self.issues[:max_issues]
        lines: list[str] = []
        for it in issues:
            loc = f" [{it.location}]" if it.location else ""
            sug = f" Suggestion: {it.suggestion}" if it.suggestion else ""
            lines.append(f"{it.severity.value.upper()} {it.code}{loc}: {it.message}{sug}")
        if max_issues is not None and len(self.issues) > max_issues:
            lines.append(f"... ({len(self.issues) - max_issues} more)")
        return "\n".join(lines)