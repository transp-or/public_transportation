from __future__ import annotations

from typing import Any

from .diagnostics import compute_all_diagnostics
from .results import MLResult


def build_ml_report_data(
    result: MLResult,
    diagnostics: dict[str, Any] | None = None,
    figure_files: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a structured report dictionary for ML results."""
    if diagnostics is None:
        diagnostics = compute_all_diagnostics(result)
    if figure_files is None:
        figure_files = {}

    metadata = diagnostics["metadata"]
    optimization = diagnostics["optimization"]
    covariance = diagnostics["covariance"]
    parameters = diagnostics["parameters"]

    return {
        "title": "Maximum Likelihood Diagnostic Report",
        "subtitle": f"Method: {metadata['method']} | Dimension: {metadata['dim']}",
        "metadata": [
            ("Method", metadata["method"]),
            ("Dimension", metadata["dim"]),
            ("Prior weight", metadata["prior_weight"]),
            ("Runtime (seconds)", round(metadata["runtime_seconds"], 3)),
            ("Timestamp", metadata["timestamp"]),
        ],
        "executive_summary": _build_executive_summary(
            optimization=optimization,
            covariance=covariance,
        ),
        "recommendations": _build_recommendations(
            optimization=optimization,
            covariance=covariance,
        ),
        "sections": [
            _build_optimization_section(optimization, figure_files),
            _build_covariance_section(covariance, figure_files),
            _build_parameter_section(parameters, figure_files),
        ],
    }


def _build_executive_summary(
    *,
    optimization: dict[str, Any],
    covariance: dict[str, Any],
) -> list[str]:
    """Build short summary bullets."""
    bullets = []
    bullets.append(
        "Optimizer converged successfully."
        if optimization["success"]
        else f"Optimizer did not report success: {optimization['message']}"
    )
    bullets.append(f"Final objective: {optimization['objective_final']:.6g}.")
    bullets.append(f"Final gradient norm: {optimization['gradient_norm']:.6g}.")
    if covariance["standard_errors_available"]:
        bullets.append("Standard errors are available.")
    else:
        bullets.append("Standard errors are not available.")
    return bullets


def _build_recommendations(
    *,
    optimization: dict[str, Any],
    covariance: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build diagnostic recommendations."""
    recommendations: list[dict[str, Any]] = []
    if not optimization["success"]:
        recommendations.append(
            {
                "level": "warning",
                "message": "The optimizer did not report convergence. Inspect the objective trace and try alternative starting values or scaling.",
            }
        )
    if not optimization["small_gradient"]:
        recommendations.append(
            {
                "level": "warning",
                "message": "The final gradient norm is not small. The solution may not be a stationary point.",
            }
        )
    if covariance["hessian_available"] and not covariance["hessian_positive_definite"]:
        recommendations.append(
            {
                "level": "warning",
                "message": "The Hessian is not positive definite. Standard errors may be unreliable.",
            }
        )
    if not recommendations:
        recommendations.append(
            {
                "level": "ok",
                "message": "No major numerical issue was detected.",
            }
        )
    return recommendations


def _build_optimization_section(
    optimization: dict[str, Any],
    figure_files: dict[str, str],
) -> dict[str, Any]:
    """Build optimization report section."""
    rows = [
        ("Success", _yes_no(optimization["success"])),
        ("Message", optimization["message"]),
        ("Iterations", optimization["num_iterations"]),
        ("Function evaluations", optimization["num_function_evaluations"]),
        ("Gradient evaluations", optimization["num_gradient_evaluations"]),
        ("Final objective", _fmt_float(optimization["objective_final"])),
        ("Best objective", _fmt_float(optimization["objective_best"])),
        ("Gradient norm", _fmt_float(optimization["gradient_norm"])),
    ]
    return {
        "title": "Optimization",
        "paragraphs": [
            "This section summarizes optimizer convergence and first-order optimality.",
        ],
        "tables": [
            {
                "columns": ["Quantity", "Value"],
                "rows": rows,
            }
        ],
        "figures": [
            {
                "title": "Objective trace",
                "filename": figure_files.get("objective_trace"),
            },
            {
                "title": "Gradient norm trace",
                "filename": figure_files.get("gradient_norm_trace"),
            },
        ],
    }


def _build_covariance_section(
    covariance: dict[str, Any],
    figure_files: dict[str, str],
) -> dict[str, Any]:
    """Build covariance/Hessian report section."""
    rows = [
        ("Hessian available", _yes_no(covariance["hessian_available"])),
        ("Hessian symmetric", _yes_no(covariance["hessian_symmetric"])),
        ("Hessian positive definite", _yes_no(covariance["hessian_positive_definite"])),
        ("Hessian min eigenvalue", _fmt_float(covariance["hessian_min_eigenvalue"])),
        ("Hessian condition number", _fmt_float(covariance["hessian_condition_number"])),
        ("Covariance available", _yes_no(covariance["covariance_available"])),
        ("Standard errors available", _yes_no(covariance["standard_errors_available"])),
        ("Invalid standard errors", _yes_no(covariance["invalid_standard_errors"])),
    ]
    return {
        "title": "Hessian and covariance",
        "paragraphs": [
            "This section summarizes whether asymptotic standard errors can be trusted.",
        ],
        "tables": [
            {
                "columns": ["Quantity", "Value"],
                "rows": rows,
            }
        ],
        "figures": [
            {
                "title": "Parameter correlation matrix",
                "filename": figure_files.get("parameter_correlation_matrix"),
            },
        ],
    }


def _build_parameter_section(
    parameters: dict[str, Any],
    figure_files: dict[str, str],
) -> dict[str, Any]:
    """Build parameter report section."""
    rows = []
    for row in parameters["parameter_summary"]:
        rows.append(
            [
                row["index"],
                row["name"],
                _fmt_float(row["estimate"]),
                _fmt_float(row["standard_error"]),
                _fmt_float(row["z_value"]),
            ]
        )

    return {
        "title": "Parameters",
        "paragraphs": [
            "This section reports point estimates and approximate standard errors when available.",
        ],
        "tables": [
            {
                "columns": ["Index", "Name", "Estimate", "Std. error", "z"],
                "rows": rows,
            }
        ],
        "figures": [
            {
                "title": "Most uncertain parameter estimates",
                "filename": figure_files.get("parameter_estimates_top"),
            },
        ],
    }


def _yes_no(value: Any) -> str:
    """Format booleans for reports."""
    if value is None:
        return "n/a"
    return "Yes" if bool(value) else "No"


def _fmt_float(value: Any) -> str:
    """Format a possibly missing number."""
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.6g}"
    except Exception:
        return str(value)
