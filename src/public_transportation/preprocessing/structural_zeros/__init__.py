"""Configuration and result contracts for structural-zero preprocessing."""

from .classification import (
    PRIMARY_REASON_PRECEDENCE,
    analyze_structural_zeros,
    classify_structural_zeros,
)
from .config import StructuralZeroConfig, load_structural_zero_config
from .errors import StructuralZeroConfigError, StructuralZeroConflictError
from .path_metrics import compute_od_path_metrics
from .persistence import StructuralZeroOutputPaths, write_structural_zero_outputs
from .progress import (
    StructuralZeroProgress,
    StructuralZeroProgressCallback,
    StructuralZeroTqdmProgress,
    structural_zero_tqdm_progress,
)
from .reconciliation import (
    FixedDemandReconciliationResult,
    load_and_reconcile_fixed_demand,
    reconcile_fixed_demand,
)
from .scenario_fingerprint import (
    fingerprint_scenario,
    scenario_fingerprint_payload_json,
)
from .service import StructuralZeroExecutionResult, run_structural_zero_preprocessing
from .topology import StructuralZeroTopology, build_structural_zero_topology
from .types import (
    ODPathMetrics,
    ODPathMetricRecord,
    ODTimeKey,
    StructuralZeroAnalysisResult,
    StructuralZeroReason,
    StructuralZeroRecord,
)

__all__ = [
    "FixedDemandReconciliationResult",
    "ODPathMetrics",
    "ODPathMetricRecord",
    "ODTimeKey",
    "PRIMARY_REASON_PRECEDENCE",
    "StructuralZeroAnalysisResult",
    "StructuralZeroConfig",
    "StructuralZeroConfigError",
    "StructuralZeroConflictError",
    "StructuralZeroReason",
    "StructuralZeroRecord",
    "StructuralZeroTopology",
    "StructuralZeroOutputPaths",
    "StructuralZeroProgress",
    "StructuralZeroProgressCallback",
    "StructuralZeroTqdmProgress",
    "StructuralZeroExecutionResult",
    "analyze_structural_zeros",
    "build_structural_zero_topology",
    "compute_od_path_metrics",
    "classify_structural_zeros",
    "fingerprint_scenario",
    "load_structural_zero_config",
    "load_and_reconcile_fixed_demand",
    "reconcile_fixed_demand",
    "run_structural_zero_preprocessing",
    "scenario_fingerprint_payload_json",
    "structural_zero_tqdm_progress",
    "write_structural_zero_outputs",
]
