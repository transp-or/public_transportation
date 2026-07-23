# Geneva GTFS example

This example validates OD estimation on a real, multi-line public-transport
timetable.  The network is formed by Geneva TPG tram lines 12, 14, and 18 on
Tuesday 2 June 2026.  The service network and timetable are real; passenger OD
demand and stop counts are synthetic so that the truth remains known.

The example implements the complete validation sequence:

1. extract a real network and timetable from an immutable Swiss GTFS snapshot;
2. postulate a time-dependent true OD matrix over four 30-minute intervals;
3. assign that matrix and generate boarding and alighting counts at every measurable stop event;
4. construct an intentionally inaccurate prior OD matrix;
5. estimate the 96 free OD/time cells by ML, MAP, or Bayesian VI;
6. reassign the estimate and compare estimated link flows with the true link flows.

## How the timetable data were obtained

The original data are the nationwide **Swiss GTFS 2026 timetable**, published
by Systemaufgaben Kundeninformation (SKI) on
[opentransportdata.swiss](https://data.opentransportdata.swiss/dataset/timetable-2026-gtfs2020).
The official catalogue is the authoritative source.  Direct automated download
from that host currently requires an API key, so the inspected immutable copy
was downloaded from the
[Mobility Database archive](https://mobilitydatabase.org/feeds/gtfs/mdb-2898),
which archives the file obtained from the official publisher.

The precise archived object was:

- file: `mdb-2898-202605290027.zip`;
- archive timestamp: 29 May 2026;
- SHA-256: `c6f06bdad9f20349ed08b45daf2ff6114f116a3c231afdd48abe80608382c5dd`;
- service date selected for the example: Tuesday 2 June 2026;
- TPG agency identifier: `881` (`Transports Publics Genevois`).

The national ZIP is not committed because it is about 155 MB compressed and
more than 2 GB uncompressed.  Instead, the repository commits the small derived
Geneva tables plus [provenance.json](data/provenance.json).  The complete,
standard-library extraction program is
[extract_geneva_gtfs.py](tools/extract_geneva_gtfs.py), so the derivation can be
audited and reproduced byte-for-byte.

The extraction keeps TPG route type `900` (the Swiss extended code for tram),
selects lines 12, 14, and 18, and retains every complete trip having at least
one departure in the half-open interval `[07:00, 09:00)`.  Platform-specific
GTFS stop IDs sharing the exact same stop name are consolidated into one
physical stop; their coordinates are averaged.  GTFS zero dwell times and
rare same-second successive events are shifted by the minimum number of seconds
needed by the strictly time-increasing assignment graph.  No trip is cropped at
the observation-window boundary.

Use of the source data is subject to the
[Swiss open-data terms](https://opentransportdata.swiss/en/terms-of-use/).
The required source attribution is: **Source: opentransportdata.swiss**.

## Scale and dimensionality

The committed scenario contains 62 physical stops, 173 trips, 4,754 stop-time
events, and four demand intervals from 07:00 to 09:00.  A dense ordered OD
matrix would contain 15,128 OD/time cells.  Only 96 timetable-feasible cells
are free; the other 15,032 are explicitly fixed to zero in `fixed_demand.csv`.
They are not parameters of ML, MAP, or VI, and the compact assignment layout
also removes their empty destination groups.

The support of the synthetic demand is deliberately known because this is a
model/code validation example, not an out-of-sample prediction exercise.  The
prior and true matrices share that support, but their temporal profiles and
magnitudes differ strongly.

The assignment produces continuous expected passenger flows. They are rounded
deterministically to the nearest integer before estimation because the
negative-binomial observation model is defined for counts. Both positive and
zero counts are retained. Only structurally impossible terminal observations
for which the graph has no boarding or alighting link are omitted.

## Running the workflow

From the repository root:

```bash
python docs/source/examples/geneva_gtfs/pre_processing/run_preprocessing.py
python docs/source/examples/geneva_gtfs/estimation/run_estimation.py --method ml
python docs/source/examples/geneva_gtfs/post_processing/run_comparison.py --method ml
```

Replace `ml` with `map` or `vi` to use the other inference engines.  ML and MAP
accept `--maxiter`; VI accepts `--vi-steps` and `--posterior-draws`.  Theta is
fixed at its known synthetic value of 5 in the initial validation experiment,
so differences reflect OD estimation rather than confounding between demand
and route-choice dispersion.

To reproduce the committed three-method benchmark, run estimation and
reassignment for `ml`, `map`, and `vi`, then generate the common report:

```bash
python docs/source/examples/geneva_gtfs/post_processing/compare_methods.py
```

The generated [method comparison](post_processing/results/method_comparison.md)
records runtime, OD error, reassigned link-flow error, optimizer termination,
and the empirical coverage and width of VI's 90% intervals. The committed
benchmark deliberately uses the inaccurate stress-test prior. It therefore
demonstrates the cost of prior misspecification; it is not evidence that
regularization is intrinsically inferior to unregularized ML.

To reproduce the committed input tables after obtaining the source archive:

```bash
python docs/source/examples/geneva_gtfs/tools/extract_geneva_gtfs.py \
  /path/to/mdb-2898-202605290027.zip
```

The extractor refuses an archive with a different checksum unless
`--skip-checksum` is explicitly supplied.
