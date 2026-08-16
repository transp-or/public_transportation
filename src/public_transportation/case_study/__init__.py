"""Retired timetable-journey case-study namespace.

The former generic case-study runner depended on the retired timetable journey
preprocessing backend. It is intentionally unavailable; retained users should
use the direct-scheduled fixed-routing APIs instead.
"""

from __future__ import annotations


class RetiredCaseStudyWorkflowError(RuntimeError):
    """Raised when the retired case-study workflow is requested."""


def require_case_study_workflow() -> None:
    """Fail clearly instead of selecting an implicit replacement backend."""
    raise RetiredCaseStudyWorkflowError(
        "The generic timetable-journey case-study workflow is retired; "
        "use the direct-scheduled fixed-routing APIs."
    )


__all__ = ["RetiredCaseStudyWorkflowError", "require_case_study_workflow"]
