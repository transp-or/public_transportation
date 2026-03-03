# src/public_transportation/inference/fingerprint_debug.py
"""
public_transportation.inference.fingerprint_debug

Small, production-quality utilities to debug fingerprint mismatches between:
- the *current* AssignmentIDManager (built from scenario + graph), and
- saved OD+theta inference results (npz), or another AssignmentIDManager.

Design goals
------------
- Pure Python (no JAX).
- Zero magic: no getattr/hasattr; work with concrete project types.
- Helpful, human-readable error messages (payload diff when available).

Prerequisites
-------------
- AssignmentIDManager stores:
    - fingerprint: str
    - fingerprint_payload_json: str
    - fingerprint_payload(): dict[str, Any]
    - format_fingerprint_mismatch(...): str

- ODThetaVIResults optionally stores:
    - fingerprint: str
    - fingerprint_payload_json: str | None
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import json

from public_transportation.assignment.id_manager import AssignmentIDManager
from public_transportation.inference.results_io import ODThetaVIResults


# -----------------------------------------------------------------------------
# Public summaries
# -----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class FingerprintSummary:
    fingerprint: str
    num_nodes: int | None
    num_links: int | None
    num_od: int | None

    def format_one_line(self) -> str:
        parts = [f"fingerprint={self.fingerprint}"]
        if self.num_nodes is not None:
            parts.append(f"num_nodes={self.num_nodes}")
        if self.num_links is not None:
            parts.append(f"num_links={self.num_links}")
        if self.num_od is not None:
            parts.append(f"num_od={self.num_od}")
        return ", ".join(parts)


def _safe_int(x: Any) -> int | None:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def summarize_id_manager(idm: AssignmentIDManager) -> FingerprintSummary:
    payload = idm.fingerprint_payload()
    return FingerprintSummary(
        fingerprint=str(idm.fingerprint),
        num_nodes=_safe_int(payload.get("num_nodes")),
        num_links=_safe_int(payload.get("num_links")),
        num_od=_safe_int(payload.get("num_od")),
    )


def summarize_results(results: ODThetaVIResults) -> FingerprintSummary:
    # Results payload JSON is optional. If absent, we at least report num_od via f0.
    num_od_from_f0 = _safe_int(results.f0.shape[0])

    if results.fingerprint_payload_json is None:
        return FingerprintSummary(
            fingerprint=str(results.fingerprint),
            num_nodes=None,
            num_links=None,
            num_od=num_od_from_f0,
        )

    try:
        payload = json.loads(results.fingerprint_payload_json)
    except Exception:
        # If payload is corrupted, still return a summary without it.
        return FingerprintSummary(
            fingerprint=str(results.fingerprint),
            num_nodes=None,
            num_links=None,
            num_od=num_od_from_f0,
        )

    if not isinstance(payload, dict):
        return FingerprintSummary(
            fingerprint=str(results.fingerprint),
            num_nodes=None,
            num_links=None,
            num_od=num_od_from_f0,
        )

    return FingerprintSummary(
        fingerprint=str(results.fingerprint),
        num_nodes=_safe_int(payload.get("num_nodes")),
        num_links=_safe_int(payload.get("num_links")),
        num_od=_safe_int(payload.get("num_od", num_od_from_f0)),
    )


def format_summary(summary: FingerprintSummary, *, title: str) -> str:
    lines: list[str] = []
    lines.append(title)
    lines.append("-" * len(title))
    lines.append(f"fingerprint: {summary.fingerprint}")
    if summary.num_nodes is not None:
        lines.append(f"num_nodes:   {summary.num_nodes}")
    if summary.num_links is not None:
        lines.append(f"num_links:   {summary.num_links}")
    if summary.num_od is not None:
        lines.append(f"num_od:      {summary.num_od}")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Debug / mismatch formatting
# -----------------------------------------------------------------------------

def format_results_vs_id_manager_mismatch(
    *,
    results: ODThetaVIResults,
    id_manager: AssignmentIDManager,
    context: str = "",
    max_list_diffs: int = 5,
) -> str:
    """Return a rich message describing why results and id_manager are incompatible."""
    exp_fp = str(results.fingerprint)
    got_fp = str(id_manager.fingerprint)

    # Prefer payload diff when results include payload JSON.
    exp_payload_json = results.fingerprint_payload_json
    got_payload_json = id_manager.fingerprint_payload_json

    # Build a compact header with summaries.
    res_sum = summarize_results(results)
    idm_sum = summarize_id_manager(id_manager)

    lines: list[str] = []
    if context:
        lines.append(f"Context: {context}")
        lines.append("")

    lines.append(format_summary(res_sum, title="Results"))
    lines.append("")
    lines.append(format_summary(idm_sum, title="Current assignment indexing"))
    lines.append("")
    lines.append(
        AssignmentIDManager.format_fingerprint_mismatch(
            expected_fingerprint=exp_fp,
            got_fingerprint=got_fp,
            expected_payload_json=exp_payload_json,
            got_payload_json=got_payload_json,
            max_list_diffs=max_list_diffs,
        )
    )
    return "\n".join(lines)


def format_id_manager_vs_id_manager_mismatch(
    *,
    expected: AssignmentIDManager,
    got: AssignmentIDManager,
    context: str = "",
    max_list_diffs: int = 5,
) -> str:
    """Return a rich message describing why two id_managers are incompatible."""
    exp_sum = summarize_id_manager(expected)
    got_sum = summarize_id_manager(got)

    lines: list[str] = []
    if context:
        lines.append(f"Context: {context}")
        lines.append("")

    lines.append(format_summary(exp_sum, title="Expected assignment indexing"))
    lines.append("")
    lines.append(format_summary(got_sum, title="Got assignment indexing"))
    lines.append("")
    lines.append(
        AssignmentIDManager.format_fingerprint_mismatch(
            expected_fingerprint=str(expected.fingerprint),
            got_fingerprint=str(got.fingerprint),
            expected_payload_json=expected.fingerprint_payload_json,
            got_payload_json=got.fingerprint_payload_json,
            max_list_diffs=max_list_diffs,
        )
    )
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Assertions (used by scripts)
# -----------------------------------------------------------------------------

def assert_results_compatible_with_id_manager(
    *,
    results: ODThetaVIResults,
    id_manager: AssignmentIDManager,
    context: str = "",
    max_list_diffs: int = 5,
) -> None:
    """Raise ValueError with a rich message if results and id_manager fingerprints do not match."""
    if str(results.fingerprint) == str(id_manager.fingerprint):
        return

    raise ValueError(
        format_results_vs_id_manager_mismatch(
            results=results,
            id_manager=id_manager,
            context=context,
            max_list_diffs=max_list_diffs,
        )
    )


def assert_id_managers_compatible(
    *,
    expected: AssignmentIDManager,
    got: AssignmentIDManager,
    context: str = "",
    max_list_diffs: int = 5,
) -> None:
    """Raise ValueError with a rich message if two id_managers fingerprints do not match."""
    if str(expected.fingerprint) == str(got.fingerprint):
        return

    raise ValueError(
        format_id_manager_vs_id_manager_mismatch(
            expected=expected,
            got=got,
            context=context,
            max_list_diffs=max_list_diffs,
        )
    )