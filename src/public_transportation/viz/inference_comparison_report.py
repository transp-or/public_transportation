"""
public_transportation.viz.inference_comparison_report

Comparison report between:
- prior OD (f0) vs estimated OD (f_hat),
- assignment link flows under f0 vs under f_hat (using the *same* theta_hat),
- predicted measurements vs observed measurements.

This module is intended for post-processing. It is pure Python/NumPy for computations
EXCEPT the assignment call, which goes through the existing adapter (JAX-based).

Important
---------
- theta_hat is REQUIRED and must be chosen by the user/script (mean/median/mode, no default here).
- This report compares measurement-space quantities to observations because observations
  live in measurement space (e.g., boardings/alightings), not on assignment links.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from public_transportation.assignment.id_manager import AssignmentIDManager
from public_transportation.measurement.mapping import AggregationSpec
from public_transportation.measurement.aggregation_numpy import (
    aggregate_link_flow_to_measurements,
    apply_detection_rate,
    as_1d_float,
)
from public_transportation.viz.html_utils import (
    KPI,
    esc,
    h1,
    h2,
    kpi_row,
    link as html_link,
    p,
    raw_p,
    table,
    table_html_cells,
    wrap_html,
    code,
)

from public_transportation.inference.assignment_adapter import build_assignment_inputs, assign_link_flow


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ODDiffRow:
    origin_stop_id: str
    dest_stop_id: str
    time_bin_index: int
    f0: float
    f_hat: float
    delta: float
    rel_delta: float  # delta / (f0 + eps)


@dataclass(frozen=True, slots=True)
class MeasurementDiffRow:
    m: int
    y_obs: float
    y_pred_prior: float
    y_pred_post: float
    err_prior: float
    err_post: float
    abs_err_prior: float
    abs_err_post: float


@dataclass(frozen=True, slots=True)
class ComparisonBundle:
    # Provenance
    fingerprint_expected: str
    fingerprint_results: str
    fingerprint_id_manager: str

    # Inputs chosen by user/script
    theta_hat: float
    rho: float

    # OD vectors
    f0: np.ndarray        # (num_od,)
    f_hat: np.ndarray     # (num_od,)

    # Link flows
    link_flow_prior: np.ndarray  # (num_links,)
    link_flow_post: np.ndarray   # (num_links,)

    # Measurements
    y_obs: np.ndarray           # (M,)
    y_pred_prior: np.ndarray    # (M,)
    y_pred_post: np.ndarray     # (M,)

    # Summaries
    od_total_f0: float
    od_total_f_hat: float
    meas_total_obs: float
    meas_total_pred_prior: float
    meas_total_pred_post: float

    rmse_prior: float
    rmse_post: float
    mae_prior: float
    mae_post: float

    # Top-k rows for display
    od_top_rows: tuple[ODDiffRow, ...]
    meas_top_rows: tuple[MeasurementDiffRow, ...]


# -----------------------------------------------------------------------------
# Core computations
# -----------------------------------------------------------------------------

def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean(d * d))) if d.size else 0.0


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(np.mean(np.abs(d))) if d.size else 0.0


def _topk_indices(values: np.ndarray, *, k: int) -> np.ndarray:
    v = np.asarray(values, dtype=float).reshape(-1)
    if v.size == 0:
        return np.zeros((0,), dtype=int)
    kk = max(0, min(int(k), int(v.size)))
    if kk == 0:
        return np.zeros((0,), dtype=int)
    # partial sort then stable sort among the selected
    idx = np.argpartition(-v, kk - 1)[:kk]
    idx = idx[np.argsort(-v[idx], kind="mergesort")]
    return idx.astype(int)


def compute_od_and_flow_comparison(
    *,
    scenario: Any,
    assignment_artifacts: Any,
    id_manager: AssignmentIDManager,
    mapping_spec: AggregationSpec,
    y_obs: np.ndarray,
    fingerprint_expected: str,
    fingerprint_results: str,
    theta_hat: float,
    f0: np.ndarray,
    f_hat: np.ndarray,
    rho: float = 1.0,
    top_k_od: int = 20,
    top_k_meas: int = 30,
    eps_rel: float = 1e-9,
) -> ComparisonBundle:
    """Compute everything needed for an OD/link/measurement comparison report.

    Parameters
    ----------
    theta_hat:
        REQUIRED. A single theta point estimate chosen by the user/script.
    f0, f_hat:
        Prior/baseline and estimated OD vectors in assignment OD indexing.
    rho:
        Detection rate used for predicted measurements (mu = rho * lambda).
    """
    # --- Fingerprints: fail fast
    if str(id_manager.fingerprint) != str(fingerprint_expected):
        raise ValueError(
            "Fingerprint mismatch between expected and id_manager: "
            f"expected={fingerprint_expected}, id_manager={id_manager.fingerprint}"
        )

    # Results fingerprint is used as a second check (npz provenance)
    if str(fingerprint_results) != str(fingerprint_expected):
        raise ValueError(
            "Fingerprint mismatch between expected and results file: "
            f"expected={fingerprint_expected}, results={fingerprint_results}"
        )

    # --- Validate theta_hat
    th = float(theta_hat)
    if not np.isfinite(th) or th <= 0.0:
        raise ValueError(f"theta_hat must be finite and > 0, got {theta_hat!r}")

    # --- Validate vectors
    f0v = as_1d_float(f0, name="f0")
    fhv = as_1d_float(f_hat, name="f_hat")
    if f0v.shape != fhv.shape:
        raise ValueError(f"f0 and f_hat shapes must match, got {f0v.shape} vs {fhv.shape}")

    y = as_1d_float(y_obs, name="y_obs")
    if int(y.shape[0]) != int(mapping_spec.num_measurements):
        raise ValueError(
            f"y_obs length {int(y.shape[0])} must match spec.num_measurements {int(mapping_spec.num_measurements)}"
        )

    # --- Assignment: build adapter inputs once
    assignment_inputs = build_assignment_inputs(artifacts=assignment_artifacts)

    # Prior run (f0, theta_hat)
    link_flow_prior = np.asarray(assign_link_flow(inputs=assignment_inputs, f=f0v, theta=th), dtype=float).reshape(-1)
    # Posterior run (f_hat, theta_hat)
    link_flow_post = np.asarray(assign_link_flow(inputs=assignment_inputs, f=fhv, theta=th), dtype=float).reshape(-1)

    if link_flow_prior.shape != link_flow_post.shape:
        raise RuntimeError("Internal error: link_flow shapes differ between runs.")

    # --- Measurement predictions (lambda then mu)
    lambda_prior = aggregate_link_flow_to_measurements(link_flow=link_flow_prior, spec=mapping_spec)
    lambda_post = aggregate_link_flow_to_measurements(link_flow=link_flow_post, spec=mapping_spec)

    y_pred_prior = apply_detection_rate(lambda_m=lambda_prior, rho=rho)
    y_pred_post = apply_detection_rate(lambda_m=lambda_post, rho=rho)

    # --- Summaries
    od_total_f0 = float(np.sum(f0v))
    od_total_f_hat = float(np.sum(fhv))

    meas_total_obs = float(np.sum(y))
    meas_total_pred_prior = float(np.sum(y_pred_prior))
    meas_total_pred_post = float(np.sum(y_pred_post))

    rmse_prior = _rmse(y, y_pred_prior)
    rmse_post = _rmse(y, y_pred_post)
    mae_prior = _mae(y, y_pred_prior)
    mae_post = _mae(y, y_pred_post)

    # --- Top-k OD differences (by absolute delta)
    delta_f = fhv - f0v
    idx_od = _topk_indices(np.abs(delta_f), k=top_k_od)

    od_rows: list[ODDiffRow] = []
    # Prefer canonical keys if you want stable, interpretable listing
    # Here: use scenario order to match indexing of f0/f_hat (assignment OD order).
    # Your AssignmentIDManager doc says OD vectors consumed in scenario order; that’s what f0 is built for.
    od_keys = id_manager.od_keys_scenario
    if len(od_keys) != int(f0v.shape[0]):
        raise ValueError(
            f"ID manager od_keys_scenario length {len(od_keys)} does not match f0 length {int(f0v.shape[0])}"
        )
    for i in idx_od.tolist():
        k = od_keys[int(i)]
        a0 = float(f0v[int(i)])
        a1 = float(fhv[int(i)])
        d = float(a1 - a0)
        rel = float(d / (a0 + float(eps_rel)))
        od_rows.append(
            ODDiffRow(
                origin_stop_id=str(k.origin_stop_id),
                dest_stop_id=str(k.dest_stop_id),
                time_bin_index=int(k.time_bin_index),
                f0=a0,
                f_hat=a1,
                delta=d,
                rel_delta=rel,
            )
        )

    # --- Top-k measurement errors (by |post error|)
    err_post = y_pred_post - y
    idx_m = _topk_indices(np.abs(err_post), k=top_k_meas)

    meas_rows: list[MeasurementDiffRow] = []
    for m_idx in idx_m.tolist():
        yo = float(y[int(m_idx)])
        yp0 = float(y_pred_prior[int(m_idx)])
        yp1 = float(y_pred_post[int(m_idx)])
        e0 = float(yp0 - yo)
        e1 = float(yp1 - yo)
        meas_rows.append(
            MeasurementDiffRow(
                m=int(m_idx),
                y_obs=yo,
                y_pred_prior=yp0,
                y_pred_post=yp1,
                err_prior=e0,
                err_post=e1,
                abs_err_prior=float(abs(e0)),
                abs_err_post=float(abs(e1)),
            )
        )

    return ComparisonBundle(
        fingerprint_expected=str(fingerprint_expected),
        fingerprint_results=str(fingerprint_results),
        fingerprint_id_manager=str(id_manager.fingerprint),
        theta_hat=th,
        rho=float(rho),
        f0=f0v,
        f_hat=fhv,
        link_flow_prior=link_flow_prior,
        link_flow_post=link_flow_post,
        y_obs=y,
        y_pred_prior=np.asarray(y_pred_prior, dtype=float),
        y_pred_post=np.asarray(y_pred_post, dtype=float),
        od_total_f0=od_total_f0,
        od_total_f_hat=od_total_f_hat,
        meas_total_obs=meas_total_obs,
        meas_total_pred_prior=meas_total_pred_prior,
        meas_total_pred_post=meas_total_pred_post,
        rmse_prior=rmse_prior,
        rmse_post=rmse_post,
        mae_prior=mae_prior,
        mae_post=mae_post,
        od_top_rows=tuple(od_rows),
        meas_top_rows=tuple(meas_rows),
    )


# -----------------------------------------------------------------------------
# HTML report writer
# -----------------------------------------------------------------------------

def write_od_theta_comparison_report_html(
    *,
    bundle: ComparisonBundle,
    output_path: str | Path,
    title: str = "OD + theta post-processing report",
    extra_links: dict[str, str] | None = None,
) -> Path:
    """Write a self-contained HTML report (tables + KPIs).

    Parameters
    ----------
    extra_links:
        Optional dict label -> href to link additional artifacts
        (e.g., time-expanded prior/post reports, PNG plots).
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    body_parts: list[str] = []
    body_parts.append(h1(title))

    # --- Provenance + checks
    if bundle.fingerprint_expected == bundle.fingerprint_id_manager == bundle.fingerprint_results:
        body_parts.append(raw_p("<div class='ok'><b>Fingerprint OK</b>: results, expected, and id_manager fingerprints match.</div>"))
    else:
        body_parts.append(
            raw_p(
                "<div class='warn'><b>Fingerprint mismatch</b>: "
                f"expected={esc(bundle.fingerprint_expected)}, "
                f"results={esc(bundle.fingerprint_results)}, "
                f"id_manager={esc(bundle.fingerprint_id_manager)}"
                "</div>"
            )
        )

    body_parts.append(
        kpi_row(
            [
                KPI("theta_hat used for assignment", f"{bundle.theta_hat:.6g}"),
                KPI("rho used for prediction", f"{bundle.rho:.6g}"),
                KPI("num_od", f"{int(bundle.f0.shape[0])}"),
                KPI("num_links", f"{int(bundle.link_flow_prior.shape[0])}"),
                KPI("num_measurements", f"{int(bundle.y_obs.shape[0])}"),
            ]
        )
    )

    if extra_links:
        items = []
        for lab, href in extra_links.items():
            items.append(f"{html_link(href, lab)}")
        body_parts.append(raw_p("<p class='muted'><b>Artifacts</b>: " + " | ".join(items) + "</p>"))

    # --- OD comparison
    body_parts.append(h2("OD comparison"))
    body_parts.append(
        kpi_row(
            [
                KPI("Total OD (prior f0)", f"{bundle.od_total_f0:.6g}"),
                KPI("Total OD (estimated f_hat)", f"{bundle.od_total_f_hat:.6g}"),
                KPI("Δ total OD", f"{(bundle.od_total_f_hat - bundle.od_total_f0):.6g}"),
            ]
        )
    )

    od_table_rows = []
    for r in bundle.od_top_rows:
        od_table_rows.append(
            [
                r.origin_stop_id,
                r.dest_stop_id,
                r.time_bin_index,
                f"{r.f0:.6g}",
                f"{r.f_hat:.6g}",
                f"{r.delta:.6g}",
                f"{100.0 * r.rel_delta:.3g}%",
            ]
        )
    body_parts.append(
        table(
            headers=["origin", "dest", "time_bin_idx", "f0", "f_hat", "Δ", "Δ/(f0)"],
            rows=od_table_rows,
            caption="Top OD changes (sorted by |Δ|).",
        )
    )

    # --- Measurement comparison (observed vs predicted)
    body_parts.append(h2("Observed vs predicted measurements"))
    body_parts.append(
        kpi_row(
            [
                KPI("Total observed", f"{bundle.meas_total_obs:.6g}"),
                KPI("Total predicted (prior OD)", f"{bundle.meas_total_pred_prior:.6g}"),
                KPI("Total predicted (estimated OD)", f"{bundle.meas_total_pred_post:.6g}"),
                KPI("RMSE (prior OD)", f"{bundle.rmse_prior:.6g}"),
                KPI("RMSE (estimated OD)", f"{bundle.rmse_post:.6g}"),
                KPI("MAE (prior OD)", f"{bundle.mae_prior:.6g}"),
                KPI("MAE (estimated OD)", f"{bundle.mae_post:.6g}"),
            ]
        )
    )

    meas_table_rows = []
    for r in bundle.meas_top_rows:
        meas_table_rows.append(
            [
                r.m,
                f"{r.y_obs:.6g}",
                f"{r.y_pred_prior:.6g}",
                f"{r.y_pred_post:.6g}",
                f"{r.err_prior:.6g}",
                f"{r.err_post:.6g}",
                f"{r.abs_err_prior:.6g}",
                f"{r.abs_err_post:.6g}",
            ]
        )
    body_parts.append(
        table(
            headers=[
                "m",
                "y_obs",
                "y_pred(prior OD)",
                "y_pred(estimated OD)",
                "err(prior)",
                "err(estimated)",
                "|err|(prior)",
                "|err|(estimated)",
            ],
            rows=meas_table_rows,
            caption="Top measurement residuals (sorted by |err_estimated|).",
        )
    )

    # --- Link-flow summary (lightweight)
    body_parts.append(h2("Link-flow summary"))
    # We do not have y_obs in link space, so we summarize prior vs post only.
    lf0 = bundle.link_flow_prior
    lf1 = bundle.link_flow_post
    dlf = lf1 - lf0

    body_parts.append(
        kpi_row(
            [
                KPI("sum link_flow (prior OD)", f"{float(lf0.sum()):.6g}"),
                KPI("sum link_flow (estimated OD)", f"{float(lf1.sum()):.6g}"),
                KPI("mean |Δ link_flow|", f"{float(np.mean(np.abs(dlf))):.6g}"),
                KPI("max |Δ link_flow|", f"{float(np.max(np.abs(dlf))):.6g}"),
            ]
        )
    )

    # Top-k links by |delta|
    k_links = min(30, int(lf0.size))
    idx_links = np.argpartition(-np.abs(dlf), k_links - 1)[:k_links] if k_links > 0 else np.zeros((0,), dtype=int)
    idx_links = idx_links[np.argsort(-np.abs(dlf[idx_links]), kind="mergesort")] if idx_links.size else idx_links

    link_rows = []
    for e in idx_links.tolist():
        link_rows.append(
            [int(e), f"{float(lf0[e]):.6g}", f"{float(lf1[e]):.6g}", f"{float(dlf[e]):.6g}", f"{float(abs(dlf[e])):.6g}"]
        )
    body_parts.append(
        table(
            headers=["link", "flow(prior OD)", "flow(estimated OD)", "Δ", "|Δ|"],
            rows=link_rows,
            caption="Top link changes (sorted by |Δ|).",
        )
    )

    # Footer / notes
    body_parts.append(
        raw_p(
            "<p class='muted'><b>Note:</b> observations come from <code>measurements_boarding_alighting.csv</code> "
            "and live in measurement space. This report compares predictions to those observations after applying "
            "the aggregation spec (and rho).</p>"
        )
    )

    html_page = wrap_html(title=title, body="\n".join(body_parts))
    out.write_text(html_page, encoding="utf-8")
    return out