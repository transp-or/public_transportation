# src/public_transportation/domain/line.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .issues import Issue, Severity, ValidationReport


@dataclass(slots=True, frozen=True)
class Line:
    """
    Public transport line (aka route).

    This corresponds to the stable service identity (e.g., "L1"), while
    individual runs are represented by `Trip` objects.

    Minimal-friction design:
    - Only `line_id` is required.
    - Everything else is optional metadata useful for reporting or extensions.

    :param line_id: Unique identifier of the line.
    :param short_name: Optional short public name (e.g., "L1", "8", "M2").
    :param long_name: Optional longer descriptive name.
    :param mode: Optional mode label (e.g., "bus", "tram", "metro").
    :param agency_id: Optional operator/agency identifier.
    :param attributes: Optional free-form dict for extra metadata.
    """

    line_id: str
    short_name: str | None = None
    long_name: str | None = None
    mode: str | None = None
    agency_id: str | None = None
    attributes: dict[str, Any] | None = None

    def validate(self) -> ValidationReport:
        """
        Validate local fields. Cross-references are validated by Timetable/Scenario.

        :return: ValidationReport describing detected issues.
        """
        rep = ValidationReport(issues=[])

        if not self.line_id:
            rep.add(
                Issue(
                    severity=Severity.ERROR,
                    code="LINE_ID_EMPTY",
                    message="line_id is empty.",
                    location="timetable.lines[].line_id",
                )
            )

        # Optional fields: validate only basic sanity (non-empty if provided).
        if self.short_name is not None and not self.short_name.strip():
            rep.add(
                Issue(
                    severity=Severity.WARNING,
                    code="LINE_SHORT_NAME_EMPTY",
                    message="short_name is blank.",
                    location=f"timetable.lines[{self.line_id}].short_name",
                )
            )

        if self.long_name is not None and not self.long_name.strip():
            rep.add(
                Issue(
                    severity=Severity.WARNING,
                    code="LINE_LONG_NAME_EMPTY",
                    message="long_name is blank.",
                    location=f"timetable.lines[{self.line_id}].long_name",
                )
            )

        if self.mode is not None and not self.mode.strip():
            rep.add(
                Issue(
                    severity=Severity.WARNING,
                    code="LINE_MODE_EMPTY",
                    message="mode is blank.",
                    location=f"timetable.lines[{self.line_id}].mode",
                )
            )

        if self.agency_id is not None and not self.agency_id.strip():
            rep.add(
                Issue(
                    severity=Severity.WARNING,
                    code="LINE_AGENCY_EMPTY",
                    message="agency_id is blank.",
                    location=f"timetable.lines[{self.line_id}].agency_id",
                )
            )

        if self.attributes is not None and not isinstance(self.attributes, dict):
            rep.add(
                Issue(
                    severity=Severity.ERROR,
                    code="LINE_ATTRIBUTES_NOT_DICT",
                    message="attributes must be a dict if provided.",
                    location=f"timetable.lines[{self.line_id}].attributes",
                    context={"type": type(self.attributes).__name__},
                )
            )

        return rep