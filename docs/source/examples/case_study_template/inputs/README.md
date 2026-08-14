# Canonical case inputs

The generic adapter accepts a strict `Scenario` directory with
`metadata.json`, `stops.csv`, `lines.csv`, `time_bins.csv`, `trips.csv`, and
`stop_times.csv`. The candidate OD universe is the pair-only
`inputs/od_pairs.csv` file with `origin_stop_id,destination_stop_id`; it is
independent of the scenario's provisional time bins. The template points at
the committed two-line scenario in `inputs/scenario/` so that it runs from a
clean public checkout. A time-dependent demand table is supported only as an
explicit legacy compatibility input.

The optional pair-level prior uses
`origin_stop_id,destination_stop_id,prior_value`. The template instead uses
`source = "all_ones"` as a neutral numerical seed. It is not observed demand,
production, or destination attractiveness.

The measurement file must contain the configured columns and identify each row
by exactly one timetable event through `trip_id` or an unambiguous `line_id`.
Missing values, duplicate keys, unknown identities, and ambiguous matches are
errors. The adapter never silently cleans or drops a row.
