"""Example 01: one OD pair, 2 lines x 2 buses = 4 equal alternatives.

This script expects the following files next to it:
- metadata.json
- stops.csv
- trips.csv
- stop_times.csv
- time_bins.csv
- demand.csv

Workflow:
1) load the Scenario from the data folder,
2) run the assignment,
3) display resulting link flows (and simple totals by link type).

Constraints:
- No new helper functions are defined in this script.
- Use only the domain/assignment API provided by the package.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp

from public_transportation.domain.scenario import Scenario
from public_transportation.assignment.config import AssignmentConfig
from public_transportation.assignment.assign import assign_from_scenario, prepare_assignment
from public_transportation.assignment.factory import build_assignment_factory
from public_transportation.viz.time_expanded_report import write_time_expanded_report_html

DATA = Path(__file__).resolve().parent / "data"

# 1) Load scenario from data
scenario = Scenario.from_folder(DATA)


# Optional: validate and print any issues
rep = scenario.validate()
if rep.issues:
    print("Scenario validation issues:")
    for it in rep.issues:
        print(f"- [{it.severity.name}] {it.code}: {it.message} ({it.location})")

# 2) Assign demand
od_values = jnp.asarray([float(r.flow) for r in scenario.demand.records], dtype=jnp.float32)
config = AssignmentConfig()

res = assign_from_scenario(
    scenario=scenario,
    od_values=od_values,
    config=config,
    theta=None,
    return_group_link_flows=False,
)

# 2b) Test the fast link_flow function (factory)
factory = build_assignment_factory(scenario=scenario, config=config)
link_flow_fast = factory.link_flow_fn(od_values, theta=res.theta)

# Sanity check: fast function must match assign_from_scenario output
if link_flow_fast.shape != res.link_flow.shape:
    raise RuntimeError(
        f"Shape mismatch: link_flow_fast {link_flow_fast.shape} vs res.link_flow {res.link_flow.shape}."
    )

# Use a tolerance suitable for float32 computations
if not bool(jnp.allclose(link_flow_fast, res.link_flow, rtol=1e-6, atol=1e-6)):
    max_abs = float(jnp.max(jnp.abs(link_flow_fast - res.link_flow)))
    raise RuntimeError(
        "Factory link_flow_fn does not match assign_from_scenario output. "
        f"Max abs diff: {max_abs}"
    )

print("Factory link_flow_fn test: OK (matches assign_from_scenario)")

# 3) Report (no extra calculations in this script)
print("Example 01 — assignment")
print(f"OD records: {len(scenario.demand.records) if scenario.demand is not None else 0}")
print(f"Theta used: {res.theta:.6g}")

arts = prepare_assignment(scenario, config)
report_path = DATA / "time_expanded_report.html"

# `write_time_expanded_report_html` is responsible for reporting exactly what the model computed.
# This script passes only scenario + graph + the computed link flows.
write_time_expanded_report_html(
    graph=arts.graph,
    output_path=report_path,
    scenario=scenario,
    link_flow=link_flow_fast,
    svg_scale_x=1.8,
    svg_scale_y=2.4,
)

print(f"\nTime-expanded graph report written to: {report_path}")
