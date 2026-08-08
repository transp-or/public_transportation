"""One-time preservation-first migration of traffic_assignment.tex.

The script reads the reviewed relocation ledger, extracts every inventoried
body range byte-for-byte, writes a hash manifest, and assembles the new source
tree with ``\\input``. It intentionally performs no editorial rewriting.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


REPORTS = Path(__file__).resolve().parent
SOURCE = REPORTS / "traffic_assignment.tex"
LEDGER = REPORTS / "traffic_assignment_reorganization_ledger.md"
ROOT = REPORTS / "traffic_assignment"

TARGET_PATHS = {
    "introduction.tex": "part1_problem_definition/introduction.tex",
    "system_and_observations.tex": "part1_problem_definition/system_and_observations.tex",
    "underdetermination.tex": "part1_problem_definition/underdetermination.tex",
    "key_modeling_questions.tex": "part1_problem_definition/key_modeling_questions.tex",
    "detailed_assignment.tex": "part2_mathematical_forward_models/detailed_assignment.tex",
    "reduced_journey_response.tex": "part2_mathematical_forward_models/reduced_journey_response.tex",
    "raptor_example.tex": "part2_mathematical_forward_models/raptor_example.tex",
    "additive_operator_decomposition.tex": "part2_mathematical_forward_models/additive_operator_decomposition.tex",
    "measurement_model.tex": "part2_mathematical_forward_models/measurement_model.tex",
    "cell_level_deviations.tex": "part3_demand_models/cell_level_deviations.tex",
    "conditional_gravity.tex": "part3_demand_models/conditional_gravity.tex",
    "route_level_ipf.tex": "part3_demand_models/route_level_ipf.tex",
    "entropy_models.tex": "part3_demand_models/entropy_models.tex",
    "model_assumptions_comparison.tex": "part3_demand_models/model_assumptions_comparison.tex",
    "map_primary_method.tex": "part4_map_and_alternatives/map_primary_method.tex",
    "maximum_likelihood_reference.tex": "part4_map_and_alternatives/maximum_likelihood_reference.tex",
    "variational_bayesian_inference.tex": "part4_map_and_alternatives/variational_bayesian_inference.tex",
    "numerical_optimization.tex": "part4_map_and_alternatives/numerical_optimization.tex",
    "method_comparison.tex": "part4_map_and_alternatives/method_comparison.tex",
    "adequacy_and_identification.tex": "part5_validation/adequacy_and_identification.tex",
    "grouped_holdout.tex": "part5_validation/grouped_holdout.tex",
    "detailed_assignment_validation.tex": "part5_validation/detailed_assignment_validation.tex",
    "advisory_relaxations.tex": "part5_validation/advisory_relaxations.tex",
    "recommended_workflow.tex": "part5_validation/recommended_workflow.tex",
    "limitations.tex": "part6_discussion/limitations.tex",
    "conclusions.tex": "part6_discussion/conclusions.tex",
    "software_architecture.tex": "appendices/software_architecture.tex",
    "indexing_contracts_and_provenance.tex": "appendices/indexing_contracts_and_provenance.tex",
    "persistence_restart_and_reporting.tex": "appendices/persistence_restart_and_reporting.tex",
    "computational_backends.tex": "appendices/computational_backends.tex",
    "scalable_linear_map.tex": "appendices/scalable_linear_map.tex",
    "stochastic_and_progressive_fidelity.tex": "appendices/stochastic_and_progressive_fidelity.tex",
    "benchmarks_and_validation_record.tex": "appendices/benchmarks_and_validation_record.tex",
    "future_extensions.tex": "appendices/future_extensions.tex",
}

# Ledger rows whose primary destination is expressed in prose rather than as a
# single backticked filename. Mixed blocks remain intact until the editorial
# split phase, as required by the preservation contract.
OVERRIDES = {
    43: "system_and_observations.tex",
    203: "system_and_observations.tex",
    336: "reduced_journey_response.tex",
    412: "reduced_journey_response.tex",
    486: "reduced_journey_response.tex",
    558: "additive_operator_decomposition.tex",
    726: "map_primary_method.tex",
    777: "detailed_assignment_validation.tex",
    806: "adequacy_and_identification.tex",
    940: "software_architecture.tex",
    995: "software_architecture.tex",
    1070: "detailed_assignment.tex",
    1259: "detailed_assignment.tex",
    1312: "detailed_assignment.tex",
    1374: "computational_backends.tex",
    1378: "software_architecture.tex",
    1402: "computational_backends.tex",
    1419: "indexing_contracts_and_provenance.tex",
    1428: "computational_backends.tex",
    1459: "additive_operator_decomposition.tex",
    1519: "detailed_assignment.tex",
    1529: "indexing_contracts_and_provenance.tex",
    1542: "software_architecture.tex",
    1636: "system_and_observations.tex",
    1658: "additive_operator_decomposition.tex",
    1682: "computational_backends.tex",
    1743: "persistence_restart_and_reporting.tex",
    1809: "map_primary_method.tex",
    1845: "scalable_linear_map.tex",
    1864: "additive_operator_decomposition.tex",
    1891: "persistence_restart_and_reporting.tex",
    1960: "computational_backends.tex",
    1971: "scalable_linear_map.tex",
    2145: "stochastic_and_progressive_fidelity.tex",
    2542: "computational_backends.tex",
    2555: "indexing_contracts_and_provenance.tex",
    2561: "measurement_model.tex",
    2610: "measurement_model.tex",
    2646: "system_and_observations.tex",
    2697: "system_and_observations.tex",
    2759: "computational_backends.tex",
    2796: "map_primary_method.tex",
    2943: "variational_bayesian_inference.tex",
}

DOCUMENT_ORDER = (
    ("Problem Definition and Identifiability", (
        "introduction.tex", "system_and_observations.tex",
        "underdetermination.tex", "key_modeling_questions.tex",
    )),
    ("Mathematical Forward Models", (
        "detailed_assignment.tex", "reduced_journey_response.tex",
        "raptor_example.tex", "additive_operator_decomposition.tex",
        "measurement_model.tex",
    )),
    ("Demand Models and Their Assumptions", (
        "cell_level_deviations.tex", "conditional_gravity.tex",
        "route_level_ipf.tex", "entropy_models.tex",
        "model_assumptions_comparison.tex",
    )),
    ("MAP Estimation and Alternative Inference Methods", (
        "map_primary_method.tex", "maximum_likelihood_reference.tex",
        "numerical_optimization.tex", "variational_bayesian_inference.tex",
        "method_comparison.tex",
    )),
    ("Validation and Model Assessment", (
        "adequacy_and_identification.tex", "grouped_holdout.tex",
        "detailed_assignment_validation.tex", "advisory_relaxations.tex",
        "recommended_workflow.tex",
    )),
    ("Limitations and Conclusions", ("limitations.tex", "conclusions.tex")),
)

APPENDIX_ORDER = (
    "software_architecture.tex",
    "indexing_contracts_and_provenance.tex",
    "persistence_restart_and_reporting.tex",
    "computational_backends.tex",
    "scalable_linear_map.tex",
    "stochastic_and_progressive_fidelity.tex",
    "benchmarks_and_validation_record.tex",
    "future_extensions.tex",
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def ledger_rows() -> list[tuple[int, int, str, str]]:
    pattern = re.compile(
        r"^\| ([0-9,]+)[–-]([0-9,]+) \| (.*?) \| (.*?) \| (.*?) \|$"
    )
    rows = []
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            rows.append(
                (
                    int(match.group(1).replace(",", "")),
                    int(match.group(2).replace(",", "")),
                    match.group(3),
                    match.group(4),
                )
            )
    return rows


def destination(start: int, primary: str) -> str:
    if start in OVERRIDES:
        return OVERRIDES[start]
    candidates = re.findall(r"`([^`]+\.tex)`", primary)
    candidates = [Path(value).name for value in candidates]
    for candidate in candidates:
        if candidate in TARGET_PATHS:
            return candidate
    raise ValueError(f"No destination for ledger row starting at {start}: {primary}")


def main() -> None:
    original = SOURCE.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)
    if len(lines) != 3064:
        raise ValueError("Source line count no longer matches the reviewed ledger.")
    rows = ledger_rows()
    coverage = [line for start, end, _, _ in rows for line in range(start, end + 1)]
    if coverage != list(range(1, 3065)):
        raise ValueError("Ledger ranges do not cover the source exactly once in order.")

    body_rows = [row for row in rows if row[0] >= 26 and row[1] <= 3058]
    preserved = ROOT / "preserved"
    preserved.mkdir(parents=True, exist_ok=True)
    target_blocks: dict[str, list[str]] = {name: [] for name in TARGET_PATHS}
    manifest_blocks = []
    reconstructed = []
    for number, (start, end, heading, primary) in enumerate(body_rows, start=1):
        content = "".join(lines[start - 1 : end])
        reconstructed.append(content)
        slug = re.sub(r"[^a-z0-9]+", "_", heading.lower()).strip("_")[:48]
        filename = f"{number:03d}_{start:04d}_{end:04d}_{slug}.tex"
        block_path = preserved / filename
        block_path.write_text(content, encoding="utf-8")
        target = destination(start, primary)
        target_blocks[target].append(filename)
        manifest_blocks.append(
            {
                "block": number,
                "source_start": start,
                "source_end": end,
                "heading": heading,
                "target": TARGET_PATHS[target],
                "preserved_file": f"preserved/{filename}",
                "sha256": digest(content),
            }
        )
    original_body = "".join(lines[25:3058])
    if "".join(reconstructed) != original_body:
        raise ValueError("Extracted blocks do not reconstruct the original body.")

    for name, relative in TARGET_PATHS.items():
        path = ROOT / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        references = target_blocks[name]
        header = (
            "% Mechanical relocation generated from the preservation ledger.\n"
            "% Existing prose is included byte-for-byte from preserved blocks.\n\n"
        )
        inputs = "".join(
            f"\\input{{traffic_assignment/preserved/{filename}}}\n"
            for filename in references
        )
        if not references:
            inputs = "% New scientific material will be added in the editorial phase.\n"
        path.write_text(header + inputs, encoding="utf-8")

    body_lines = [
        "% Modular body generated by the preservation-first migration.\n",
    ]
    for part_index, (title, names) in enumerate(DOCUMENT_ORDER):
        if part_index:
            body_lines.append("\n\\clearpage\n")
        body_lines.append(f"\n\\part{{{title}}}\n")
        for name in names:
            body_lines.append(f"\\input{{traffic_assignment/{TARGET_PATHS[name]}}}\n")
    body_lines.append("\n\\appendix\n\\clearpage\n")
    body_lines.append("\\part{Implementation and Validation Appendices}\n")
    for name in APPENDIX_ORDER:
        body_lines.append(f"\\input{{traffic_assignment/{TARGET_PATHS[name]}}}\n")
    (ROOT / "document_body.tex").write_text("".join(body_lines), encoding="utf-8")

    migrated = "".join(lines[:25])
    migrated += "\\input{traffic_assignment/document_body}\n\n"
    migrated += "".join(lines[3058:])
    SOURCE.write_text(migrated, encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "source_revision": "55ed81601d0d786f80b333f4e2b0474505c6a472",
        "source_lines": 3064,
        "source_sha256": digest(original),
        "original_body_sha256": digest(original_body),
        "blocks": manifest_blocks,
    }
    (ROOT / "relocation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
