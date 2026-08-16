# Canonical case inputs

The generic adapter accepts a strict `Scenario` directory with the fixed
stems `metadata.json`, `stops`, `lines`, and `time_bins`; the tabular stems may
use `.csv`, `.parquet`, or `.json`. `trips` and `stop_times` are optional, but
must be supplied together when timetable routing is needed. For example, the
template uses `metadata.json`, `stops.csv`, `lines.csv`, `time_bins.csv`,
`trips.csv`, and `stop_times.csv` under `inputs/scenario/`; the directory is
selected by `[paths].scenario_directory`, so these are not universal paths.
The candidate OD universe is the pair-only file configured by
`[paths].od_pairs` (the template example is `inputs/od_pairs.csv`) and is
independent of the scenario's provisional time bins. The measurement file is
selected separately by `[paths].measurements` (the template example is
`inputs/measurements_boarding_alighting.csv`). A time-dependent demand table
is supported only as an explicit legacy compatibility input and is not
required for the independent OD workflow.

The optional pair-level prior uses
`origin_stop_id,destination_stop_id,prior_value`. The template instead uses
`source = "all_ones"` as a neutral numerical seed. It is not observed demand,
production, or destination attractiveness.

The measurement file must contain the configured columns and identify each row
by exactly one timetable event through `trip_id` or an unambiguous `line_id`.
Missing values, duplicate keys, unknown identities, and ambiguous matches are
errors. The adapter never silently cleans or drops a row.
