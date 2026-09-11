"""Input normalization and provenance helpers for residual audits."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .model import ResidualAuditRun


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv_table(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"input table does not exist: {source}")
    try:
        return pd.read_csv(source)
    except Exception as error:  # pandas exposes several parser exception types.
        raise ValueError(f"could not read CSV table {source}: {error}") from error


def _check_unique_ids(frame: pd.DataFrame, *, table_name: str) -> pd.DataFrame:
    if "observation_id" not in frame.columns:
        if "row_index" in frame.columns:
            frame = frame.copy()
            if "source_row_index" not in frame.columns:
                frame["source_row_index"] = frame["row_index"]
            frame = frame.rename(columns={"row_index": "observation_id"})
        else:
            raise ValueError(
                f"{table_name} is missing required column 'observation_id' "
                "(or the persisted-report alias 'row_index')."
            )
    if frame["observation_id"].isna().any():
        raise ValueError(f"{table_name} contains missing observation_id values.")
    if frame["observation_id"].duplicated().any():
        raise ValueError(f"{table_name} contains duplicate observation_id values.")
    frame = frame.copy()
    frame["__audit_id"] = frame["observation_id"].map(_id_key)
    if frame["__audit_id"].duplicated().any():
        raise ValueError(f"{table_name} contains duplicate canonical observation IDs.")
    return frame


def _id_key(value: object) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _rename_aliases(frame: pd.DataFrame, *, table_name: str) -> pd.DataFrame:
    aliases = {
        "observed": "observed_value",
        "modeled": "predicted_value",
        "predicted": "predicted_value",
        "mean": "predicted_value",
        "residual": "raw_residual_input",
        "standardized_residual": "standardized_residual_input",
    }
    frame = frame.copy()
    for source, target in aliases.items():
        if target not in frame.columns and source in frame.columns:
            frame = frame.rename(columns={source: target})
    return _check_unique_ids(frame, table_name=table_name)


def normalize_observation_table(table: pd.DataFrame, *, table_name: str = "observation table") -> pd.DataFrame:
    """Normalize an observation table while retaining arbitrary metadata columns."""
    frame = _rename_aliases(table, table_name=table_name)
    missing = sorted({"observed_value", "predicted_value"} - set(frame.columns))
    if missing:
        raise ValueError(f"{table_name} is missing required column(s): {', '.join(missing)}")
    return frame


def normalize_contribution_table(table: pd.DataFrame) -> pd.DataFrame:
    frame = _check_unique_ids(table, table_name="contribution table")
    if len(frame) == 0:
        return frame
    known = {"od_contribution", "initial_onboard_contribution", "terminal_outflow_contribution", "fixed_offset", "scaled_mean"}
    aliases = {
        "od_flow": "od_contribution",
        "initial_onboard_flow": "initial_onboard_contribution",
        "terminal_outflow_flow": "terminal_outflow_contribution",
        "modeled_mean": "scaled_mean",
    }
    for source, target in aliases.items():
        if target not in frame.columns and source in frame.columns:
            frame = frame.rename(columns={source: target})
    numeric = [column for column in frame.columns if column in known or column.endswith("_flow")]
    for column in numeric:
        try:
            values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(f"contribution column {column!r} must be numeric.") from error
        if not np.all(np.isfinite(values)):
            raise ValueError(f"contribution column {column!r} must be finite.")
        frame[column] = values
    return frame


def normalize_metadata_table(table: pd.DataFrame) -> pd.DataFrame:
    frame = _check_unique_ids(table, table_name="metadata table")
    return frame


def join_metadata(observations: pd.DataFrame, metadata: pd.DataFrame | None) -> pd.DataFrame:
    if metadata is None:
        return observations.copy()
    if set(observations["__audit_id"]) != set(metadata["__audit_id"]):
        missing = sorted(set(observations["__audit_id"]) - set(metadata["__audit_id"]))[:5]
        extra = sorted(set(metadata["__audit_id"]) - set(observations["__audit_id"]))[:5]
        raise ValueError(
            "metadata table cannot be joined to observations; IDs differ "
            f"(missing examples={missing}, extra examples={extra})."
        )
    metadata_columns = [column for column in metadata.columns if column not in {"observation_id", "__audit_id"}]
    overlap = set(metadata_columns) & set(observations.columns)
    # Persisted prediction tables may already carry the same metadata columns;
    # keep the primary table's values and use only genuinely new metadata.
    metadata_columns = [column for column in metadata_columns if column not in overlap]
    return observations.merge(
        metadata[["__audit_id", *metadata_columns]],
        on="__audit_id",
        how="left",
        sort=False,
        validate="one_to_one",
    )


def _merge_optional_table(observations: pd.DataFrame, table: pd.DataFrame, *, table_name: str) -> pd.DataFrame:
    if set(observations["__audit_id"]) != set(table["__audit_id"]):
        raise ValueError(f"{table_name} cannot be joined: observation IDs differ.")
    values = table.drop(columns=["observation_id"], errors="ignore")
    overlapping = set(values.columns) & set(observations.columns) - {"__audit_id"}
    if overlapping:
        # Persisted residual artifacts commonly repeat the observation metadata
        # as well as observed/predicted values.  The primary prediction table
        # is authoritative; repeated columns are discarded after the IDs have
        # been checked, while genuinely new residual columns are retained.
        values = values.drop(columns=sorted(overlapping))
    return observations.merge(values, on="__audit_id", how="left", sort=False, validate="one_to_one")


def load_run(
    *,
    predicted_measurements: str | Path,
    residuals: str | Path | None = None,
    contributions: str | Path | None = None,
    metadata: str | Path | None = None,
    model_manifest: str | Path | None = None,
) -> ResidualAuditRun:
    """Load and normalize one persisted run without running any model code."""
    predicted_path = Path(predicted_measurements)
    observations = normalize_observation_table(read_csv_table(predicted_path), table_name="predicted measurements")
    input_paths: dict[str, str] = {"predicted_measurements": str(predicted_path.resolve())}
    input_fingerprints = {"predicted_measurements": sha256_file(predicted_path)}
    if residuals is not None:
        residual_path = Path(residuals)
        residual_table = _rename_aliases(read_csv_table(residual_path), table_name="residuals")
        observations = _merge_optional_table(observations, residual_table, table_name="residuals")
        input_paths["residuals"] = str(residual_path.resolve())
        input_fingerprints["residuals"] = sha256_file(residual_path)
    contribution_table = None
    if contributions is not None:
        contribution_path = Path(contributions)
        contribution_table = normalize_contribution_table(read_csv_table(contribution_path))
        if len(contribution_table) == 0:
            contribution_table = None
        elif set(observations["__audit_id"]) != set(contribution_table["__audit_id"]):
            raise ValueError("contribution table cannot be joined: observation IDs differ.")
        input_paths["contributions"] = str(contribution_path.resolve())
        input_fingerprints["contributions"] = sha256_file(contribution_path)
    metadata_table = None
    if metadata is not None:
        metadata_path = Path(metadata)
        metadata_table = normalize_metadata_table(read_csv_table(metadata_path))
        if len(metadata_table) == 0:
            metadata_table = None
        input_paths["metadata"] = str(metadata_path.resolve())
        input_fingerprints["metadata"] = sha256_file(metadata_path)
    observations = join_metadata(observations, metadata_table)
    model_fingerprints: list[str] = []
    specification_fingerprints: list[str] = []
    provenance: dict[str, Any] = {}
    if model_manifest is not None:
        manifest_path = Path(model_manifest)
    else:
        candidate = predicted_path.parent / "report.json"
        manifest_path = candidate if candidate.is_file() else None
    if manifest_path is not None and Path(manifest_path).is_file():
        try:
            payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"could not read model manifest {manifest_path}: {error}") from error
        if isinstance(payload, Mapping):
            for key in ("model_fingerprint", "model_identity_fingerprint"):
                value = payload.get(key)
                if value is not None:
                    model_fingerprints.append(str(value))
            for key in ("specification_fingerprint",):
                value = payload.get(key)
                if value is not None:
                    specification_fingerprints.append(str(value))
            nested = payload.get("provenance")
            if isinstance(nested, Mapping):
                for key in ("model_fingerprint", "model_identity_fingerprint", "specification_fingerprint"):
                    value = nested.get(key)
                    if value is not None:
                        (model_fingerprints if key != "specification_fingerprint" else specification_fingerprints).append(str(value))
            input_paths["model_manifest"] = str(Path(manifest_path).resolve())
            input_fingerprints["model_manifest"] = sha256_file(manifest_path)
            provenance = dict(payload)
    return ResidualAuditRun(
        observations=observations,
        contributions=contribution_table,
        metadata=metadata_table,
        input_paths=input_paths,
        input_fingerprints=input_fingerprints,
        model_fingerprints=tuple(dict.fromkeys(model_fingerprints)),
        specification_fingerprints=tuple(dict.fromkeys(specification_fingerprints)),
        provenance=provenance,
    )
