"""Configuration-driven case-study orchestration.

The public helpers in this package deliberately stop at the boundary between
canonical case inputs and case-specific scientific decisions.  A case can use
the generic adapter and runner when its files follow the documented canonical
formats; genuinely non-standard input transformations remain an explicit hook
in the case repository.
"""

from .config import (
    CASE_CONFIG_SCHEMA_VERSION,
    CaseStudyConfig,
    CaseStudyConfigError,
    ExpansionSettings,
    load_case_study_config,
)
from .adapter import (
    GenericCaseAdapter,
    GenericCaseBaseData,
    GenericCaseData,
    GenericCaseAudit,
    GenericCaseHook,
    load_canonical_measurements,
)
from .runner import GenericCaseRunner, run_case_stage

__all__ = [
    "CASE_CONFIG_SCHEMA_VERSION",
    "CaseStudyConfig",
    "CaseStudyConfigError",
    "ExpansionSettings",
    "GenericCaseAdapter",
    "GenericCaseAudit",
    "GenericCaseData",
    "GenericCaseBaseData",
    "GenericCaseHook",
    "GenericCaseRunner",
    "load_canonical_measurements",
    "load_case_study_config",
    "run_case_stage",
]
