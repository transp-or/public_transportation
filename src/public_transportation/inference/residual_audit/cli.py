"""Command-line interface for persisted residual audits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .io import load_run
from .model import ResidualAuditConfig
from .report import write_residual_audit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze persisted predicted measurements without rerunning assignment, "
            "preparation, or fitting."
        )
    )
    parser.add_argument("--predicted-measurements", type=Path)
    parser.add_argument("--residuals", type=Path)
    parser.add_argument("--contributions", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--model-manifest", type=Path)
    parser.add_argument("--run-a", type=Path, help="directory containing run A artifacts")
    parser.add_argument("--run-b", type=Path, help="directory containing run B artifacts")
    parser.add_argument("--allow-subset-comparison", action="store_true")
    parser.add_argument("--group-by", action="append", default=[])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, help="optional JSON or TOML configuration")
    parser.add_argument("--positive-observation-threshold", type=float, default=None)
    parser.add_argument("--predicted-mean-floor", type=float, default=None)
    parser.add_argument("--pearson-threshold", type=float, default=None)
    parser.add_argument("--variance-floor", type=float, default=None)
    parser.add_argument("--relative-observation-threshold", type=float, default=None)
    parser.add_argument("--is-holdout-evaluation", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def _config_payload(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        if path.suffix.lower() == ".toml":
            import tomllib

            return dict(tomllib.loads(path.read_text(encoding="utf-8")))
        return dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read residual-audit configuration {path}: {error}") from error


def _run_directory(path: Path, *, metadata: Path | None = None) -> dict[str, Path | None]:
    if not path.is_dir():
        raise ValueError(f"run directory does not exist: {path}")
    def optional(name: str) -> Path | None:
        candidate = path / name
        return candidate if candidate.is_file() else None
    predicted = path / "predicted_measurements.csv"
    if not predicted.is_file():
        raise ValueError(f"run directory is missing {predicted}")
    return {
        "predicted_measurements": predicted,
        "residuals": optional("residuals.csv"),
        "contributions": optional("measurement_contributions.csv"),
        "metadata": metadata,
        "model_manifest": optional("report.json"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config_payload = _config_payload(args.config)
        threshold_payload = config_payload.get("thresholds", config_payload)
        if not isinstance(threshold_payload, dict):
            raise ValueError("residual-audit configuration thresholds must be an object.")
        threshold_names = (
            "positive_observation_threshold",
            "predicted_mean_floor",
            "pearson_threshold",
            "variance_floor",
            "relative_observation_threshold",
        )
        values = {name: threshold_payload[name] for name in threshold_names if name in threshold_payload}
        for name in threshold_names:
            cli_value = getattr(args, name)
            if cli_value is not None:
                values[name] = cli_value
        config = ResidualAuditConfig(**values)
        if args.run_a is not None or args.run_b is not None:
            if args.run_a is None or args.run_b is None:
                raise ValueError("--run-a and --run-b must be supplied together.")
            run_a_paths = _run_directory(args.run_a, metadata=args.metadata)
            run_b_paths = _run_directory(args.run_b, metadata=args.metadata)
            run = load_run(**run_a_paths)
            comparison = load_run(**run_b_paths)
        else:
            if args.predicted_measurements is None:
                raise ValueError("--predicted-measurements is required unless --run-a/--run-b are used.")
            run = load_run(
                predicted_measurements=args.predicted_measurements,
                residuals=args.residuals,
                contributions=args.contributions,
                metadata=args.metadata,
                model_manifest=args.model_manifest,
            )
            comparison = None
        write_residual_audit(
            run,
            output_directory=args.output,
            group_by=args.group_by,
            config=config,
            comparison_run=comparison,
            allow_subset_comparison=args.allow_subset_comparison,
            is_holdout_evaluation=args.is_holdout_evaluation,
            force=args.force,
        )
        return 0
    except (OSError, TypeError, ValueError, KeyError) as error:
        print(f"residual audit failed: {error}", file=sys.stderr)
        return 2
