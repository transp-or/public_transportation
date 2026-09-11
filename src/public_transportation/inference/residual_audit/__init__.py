"""Reusable residual analysis for persisted gravity-model artifacts."""

from .comparison import ResidualRunComparison, compare_runs
from .grouping import GROUP_METRIC_COLUMNS, summarize_by, summarize_comparison_by
from .io import (
    join_metadata,
    load_run,
    normalize_contribution_table,
    normalize_metadata_table,
    normalize_observation_table,
    read_csv_table,
    sha256_file,
)
from .metrics import (
    compute_residual_diagnostics,
    compute_residual_metrics,
    poisson_deviance_residual,
)
from .model import ResidualAuditConfig, ResidualAuditResult, ResidualAuditRun
from .report import (
    RESIDUAL_AUDIT_SCHEMA_VERSION,
    audit_residuals,
    audit_run,
    write_audit,
    write_residual_audit,
)

__all__ = [
    "GROUP_METRIC_COLUMNS",
    "RESIDUAL_AUDIT_SCHEMA_VERSION",
    "ResidualAuditConfig",
    "ResidualAuditResult",
    "ResidualAuditRun",
    "ResidualRunComparison",
    "audit_run",
    "audit_residuals",
    "compare_runs",
    "compute_residual_diagnostics",
    "compute_residual_metrics",
    "join_metadata",
    "load_run",
    "normalize_contribution_table",
    "normalize_metadata_table",
    "normalize_observation_table",
    "poisson_deviance_residual",
    "read_csv_table",
    "sha256_file",
    "summarize_by",
    "summarize_comparison_by",
    "write_residual_audit",
    "write_audit",
]
