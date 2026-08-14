# Canonical case inputs

The generic adapter accepts a strict `Scenario` directory with
`metadata.json`, `stops.csv`, `lines.csv`, `time_bins.csv`, `trips.csv`, and
`stop_times.csv`, plus candidate demand with
`origin_stop_id,dest_stop_id,time_bin_id,flow`. The template points at the
committed two-line scenario in `inputs/scenario/` so that it runs from a clean
public checkout.

The measurement file must contain the configured columns and identify each row
by exactly one timetable event through `trip_id` or an unambiguous `line_id`.
Missing values, duplicate keys, unknown identities, and ambiguous matches are
errors. The adapter never silently cleans or drops a row.
