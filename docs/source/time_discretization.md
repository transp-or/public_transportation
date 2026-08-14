# Data-driven time discretization

`public_transportation.preprocessing.time_discretization` is a preparation-stage
diagnostic. It profiles timestamped boarding and alighting observations, detects
stable high-activity intervals, and compares uniform and peak-adaptive time-bin
schemes.

It does not modify a scenario or overwrite `time_bins.csv`. Execution produces a
JSON recommendation that can be reviewed before the selected edges are adopted
by a data-preparation pipeline.

After review, use the separate materializer to make the explicit handoff to a
scenario. It validates the selected candidate and writes the canonical
three-column CSV atomically; it refuses to replace an existing file unless
`--overwrite` is requested:

```bash
uv run python -m public_transportation.preprocessing.materialize_time_bins \
  --recommendation-json time_discretization_recommendation.json \
  --output inputs/time_bins.csv
```

To choose a named candidate instead of the report's default, add for example
`--candidate peak_adaptive`. The resulting IDs must also be used by the
scenario's demand rows through `time_bin_id`. Changing the bins requires the
downstream structural-zero, OD-candidate, reduced-journey, and response/operator
preparation stages to be rerun; the JSON report is provenance, not an
estimation input.

```bash
uv run python -m public_transportation.preprocessing.time_discretization \
  --measurements measurements_boarding_alighting.csv \
  --base-resolution-minutes 5 \
  --min-bin-minutes 10 \
  --max-bin-minutes 60 \
  --max-bins 24 \
  --num-od-pairs 1000 \
  --max-od-cells 24000 \
  --output-json time_discretization_recommendation.json
```

The input must retain timestamps at the resolution at which peaks are to be
detected. If observations have already been aggregated into wide intervals, the
tool reports that finer temporal variation cannot be recovered.

The JSON document contains the detected peak intervals, every candidate scheme,
within-bin deviance, event counts per bin, optional OD-cell estimates, and a
recommended list of half-open `time_bins`. When `num_od_pairs` and
`max_od_cells` are supplied, candidates exceeding the OD/time-cell budget are
rejected. The recommendation is deliberately
separate from scenario mutation so that a reviewer can compare it with a fixed
complexity and memory budget first.
