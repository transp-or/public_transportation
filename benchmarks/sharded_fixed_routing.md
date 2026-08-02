# Sharded fixed-routing benchmark

The deterministic Simple Example 02 benchmark prepared seven destination
groups and 5,846 links in four shards of at most two groups. The padded final
shard reused the same compiled routing kernel, for one compilation in total.

Complete and sharded routing masks agreed exactly. The maximum probability
difference was zero. The persisted cache occupied 212,754 bytes and a second
preparation reused all four shards in approximately 0.0032 seconds. No global
measurement-by-OD matrix was constructed.

The recorded timings are descriptive and machine-specific. CI checks structural
and numerical invariants rather than wall-clock thresholds. Full details are in
`sharded_fixed_routing.json`.
