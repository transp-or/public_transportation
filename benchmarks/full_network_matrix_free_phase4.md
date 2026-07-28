# Full-network matrix-free feasibility: Phase 4

## Outcome

This phase profiles and replaces the full-network strict measurement resolver,
characterizes the boarding/alighting structure, and adds an exact compact
event-aligned aggregation prototype.

## Resolver diagnosis and correction

The former strict resolver found the contributing links for every measurement
with `np.where` over all graph links. On the TPG full network this meant up to
426,436 repeated scans of 1,456,458 links. The measured strict-mapping stage was
approximately 121.2 s.

The new resolver builds two node-keyed CSR indices once: departure event to
incoming access links, and arrival event to outgoing egress links. Strict stop,
trip/line, time, event-kind, uniqueness, and nonempty-link checks are unchanged.
The CSR result is tested against the legacy full-link scan for every node in a
synthetic graph.

On the full-network scenario, synchronized mapping now takes 2.312 s:

| Phase | Seconds |
|---|---:|
| Event and link CSR construction | 0.242 |
| 426,436 strict record resolutions | 1.642 |
| Mapping arrays and report entries | 0.422 |
| Device synchronization | 0.000020 |
| Total | 2.312 |

This is approximately a 52-fold reduction from the previous 121.2 s. Scenario,
assignment, ID-manager, and CSV preparation are reported separately in
`benchmarks/full_network_measurement_mapping.json`.

## Measurement structure

The 426,436 rows produce only 431,849 link contributions:

- all 213,218 alighting measurements map to exactly one egress link;
- 207,805 boarding measurements map to one access link;
- 5,413 boarding measurements map to two access links;
- no measurement maps to more than two links.

The two-link boarding cases arise because the graph can provide two admissible
time-bin access links entering the same departure event. Collapsing every row
to one link would therefore change the established strict semantics.

## Event-aligned representation

`EventAlignedAggregationSpec` stores one primary link index per measurement and
only the 5,413 secondary boarding measurement/link pairs. Prediction is one
direct gather followed by a small scatter-add. Conversion rejects missing rows
or mappings with more than two contributions rather than silently losing data.

| Representation | Compile (s) | Warm aggregation (s) | Index storage |
|---|---:|---:|---:|
| Generic measurement/link scatter | 0.0404 | 0.000482 | 3,454,792 bytes |
| Event-aligned primary + secondary | 0.0281 | 0.000108 | 1,749,048 bytes |

Predictions were bit-identical. Focused tests also compare the complete
link-flow gradient and obtain exact equality.

## Decision

The CSR resolver should replace the repeated link scans unconditionally because
it preserves semantics and removes a material startup bottleneck. The compact
event-aligned representation is useful for full-network storage and aggregation,
but its roughly 0.37 ms saving is negligible beside the six-second destination
loader. It remains an explicit prepared representation rather than adding
automatic pipeline complexity at this stage.

The next performance phase must address the loader itself: rematerialization or
an explicit adjoint, and a process-level destination parallelism/memory model.
