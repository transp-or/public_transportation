"""
Load the synthetic case-study data into the domain data structures and validate it.

Usage
-----
python load_model.py /path/to/case_study_small

Notes
-----
- This script uses `Scenario.from_folder(...)` and `Scenario.validate()`.
- Your current `Scenario.from_folder()` (as last patched) loads trips + stop_times,
  but it does NOT yet read the `capacity` column from trips.csv. So Trip.capacity
  will remain None unless you update the loader accordingly. Validation still works.

This script can also render plots of the scenario.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from public_transportation.domain import Scenario, Severity
from public_transportation.viz import render_scenario, RenderOptions, RenderOutputs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Load and validate a public_transportation domain scenario folder."
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Path to the case-study folder containing metadata.json, stops.*, time_bins.*, demand.*, trips.*, stop_times.*",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not display figures on screen (useful on headless systems).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="If provided, save generated figures as PNG files into this directory.",
    )
    parser.add_argument(
        "--basename",
        type=str,
        default="scenario",
        help="Base name for saved figure files (without extension).",
    )
    parser.add_argument(
        "--show-all-trips",
        action="store_true",
        help="Overlay all trips on the map (default shows one representative trip per line-direction).",
    )
    parser.add_argument(
        "--no-demand",
        action="store_true",
        help="Do not generate the demand map.",
    )
    args = parser.parse_args()

    folder: Path = args.folder
    if not folder.exists() or not folder.is_dir():
        print(f"ERROR: folder does not exist or is not a directory: {folder}", file=sys.stderr)
        return 2

    # Load
    try:
        scenario = Scenario.from_folder(folder)
    except Exception as e:
        print(f"ERROR while loading scenario: {e}", file=sys.stderr)
        return 2

    # Validate
    report = scenario.validate()

    # Print summary
    n_err = sum(1 for i in report.issues if i.severity == Severity.ERROR)
    n_warn = sum(1 for i in report.issues if i.severity == Severity.WARNING)
    n_info = sum(1 for i in report.issues if i.severity == Severity.INFO)

    print(f"Loaded scenario: {scenario.metadata.title if hasattr(scenario, 'metadata') else '<no metadata>'}")
    print(f"Stops:      {len(scenario.stops)}")
    print(f"Time bins:  {len(scenario.time_bins)}")
    print(f"Demand rec: {len(scenario.demand.records) if hasattr(scenario.demand, 'records') else 'n/a'}")
    # Removed line printing links count because Scenario no longer has links attribute
    # print(f"Links:      {len(scenario.links)}")
    print(f"Timetable:  {'present' if scenario.timetable is not None else 'absent'}")
    if scenario.timetable is not None:
        print(f"  Trips:       {len(scenario.timetable.trips)}")
        print(f"  Stop times:  {len(scenario.timetable.stop_times)}")

    print()
    print(f"Validation issues: {n_err} error(s), {n_warn} warning(s), {n_info} info")

    if report.issues:
        print("\nDetailed issues:")
        # stable ordering: ERROR -> WARNING -> INFO
        order = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}
        for iss in sorted(report.issues, key=lambda x: (order.get(x.severity, 9), x.code, x.location or "")):
            loc = f" @ {iss.location}" if iss.location else ""
            sug = f" | suggestion: {iss.suggestion}" if iss.suggestion else ""
            ctx = f" | context: {iss.context}" if getattr(iss, "context", None) else ""
            print(f"- {iss.severity.name}: {iss.code}{loc}: {iss.message}{sug}{ctx}")

    if n_err == 0:
        try:
            _ = render_scenario(
                scenario,
                outputs=RenderOutputs(
                    network_map=True,
                    demand_map=(not args.no_demand),
                    timetable_panel=True,
                    space_time=True,
                ),
                options=RenderOptions(
                    show_all_trips=args.show_all_trips,
                    basename=args.basename,
                ),
                display=(not args.no_display),
                out_dir=args.out_dir,
            )
        except Exception as e:
            print(f"ERROR while rendering scenario: {e}", file=sys.stderr)
            return 2

    # Exit code: 0 if no errors, else 1
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())