# TPG assignment-preparation profile and persistent cache

Environment: TPG `two_lines_morning_time`, default `AssignmentConfig`, JAX/JAXLIB
0.8.1 CPU on Apple arm64, cached 3,690 x 15,772 measurement operator, and the
persistent objective compilation cache. The TPG repository was read and executed
without source modification.

## Root cause and internal profile

The original `prepare_assignment` time was 3.313 s. Instrumentation localized
3.142 s to destination/OD grouping. The grouping builder repeatedly evaluated
`int(graph.node_kind[i])` and related scalar expressions while scanning JAX
arrays from Python. Each scalar read could synchronize with the device.

The builder now obtains one NumPy view of immutable node metadata and uses it for
all indexing and validation. Destination link masks are constructed with one
broadcasted NumPy expression. Semantics and stable destination ordering are
unchanged. After this focused change, uncached artifact construction is 0.097 s
in the fresh-process benchmark. A synchronized detailed refresh measured:

| Stage | Seconds | Calls |
|---|---:|---:|
| Input/configuration validation | 0.000011 | 1 |
| Stop/trip/line/timetable/time-bin indexing and nodes | 0.015663 | 1 |
| Link creation and concatenation | 0.018631 | 1 |
| Link/node canonical ordering | 0.000317 | 1 |
| CSR adjacency | 0.000377 | 1 |
| Padded adjacency and masks | 0.008411 | 1 |
| Graph metadata/report construction | 0.000014 | 1 |
| Graph NumPy-to-JAX transfer and synchronization | 0.005401 | 1 |
| OD and destination indexing | 0.019716 | 1 |
| Stable destination grouping | 0.000714 | 1 |
| Padded OD indices/masks | 0.000114 | 1 |
| Destination link masks | 0.000521 | 1 |
| OD NumPy-to-JAX transfer and synchronization | 0.000426 | 1 |
| Cost arrays and synchronization | 0.079195 | 1 |

The strict scenario load, parsing, and domain validation remain outside
`prepare_assignment` and took 0.338--0.359 s. No second full `Scenario.validate`
pass is performed during assignment preparation. Graph construction validates
nonempty counts, topology inputs, indices, and destination node kinds as part of
the corresponding stages above.

The artifact contains 36,478 nodes, 11,259 links, 22,022 OD records, and 113
destination groups. Explicit graph, grouping, and cost arrays occupy 6,909,418
logical bytes. The compressed cache occupies 576,227 bytes. The machine-readable
benchmark contains every array's shape, dtype, and byte count.

## Redundant work audit

- `build_od_groups` previously converted or synchronized graph scalar values
  thousands of times; this was removed.
- `AssignmentIDManager.build` still creates OD lookup dictionaries, canonical OD
  permutations, and NumPy graph views. It takes about 0.031 s and supplies
  reporting/mapping semantics not present in the assignment kernel, so it was not
  folded into the cache.
- `build_assignment_inputs` normalizes dtypes and computes generalized access
  costs. Cached arrays already arrive with stable dtypes; JAX generally reuses
  their device buffers.
- Compact OD grouping is layout-dependent and therefore remains outside the
  demand-independent assignment artifact.
- Measurement mapping is observation-dependent and remains separate.
- Measurement-operator provenance hashes the compact assignment inputs. It does
  not rebuild the graph, but retaining this independent content hash prevents an
  assignment cache from incorrectly validating an incompatible operator.

## Cache schema and policy

`prepare_assignment` remains uncached by default. A caller can pass
`cache_directory` and `cache_policy`, or set
`PUBLIC_TRANSPORTATION_ASSIGNMENT_CACHE_DIR` and
`PUBLIC_TRANSPORTATION_ASSIGNMENT_CACHE_POLICY`. Policies are `off`, `auto`,
`refresh`, and `readonly`.

The cache is compressed NPZ with canonical JSON metadata, never pickle. It
stores graph arrays and labels, OD grouping arrays, cost arrays, counts, an array
manifest, the complete provenance payload, package version, and schema version.
The provenance covers all stops, lines, trips, stop times, time bins, OD keys,
configuration/cost constants, dtype, sentinel conventions, and indexing rules.
Demand magnitudes, observations, priors, optimizer settings, and iteration count
are intentionally excluded. Writes use a unique temporary file, `fsync`, and
atomic replacement, allowing concurrent writers of the same deterministic entry.

Loads validate the key, exact provenance JSON, schema and package versions,
recognized array names, shapes, dtypes, node/link/OD/group counts, CSR and padded
mask relationships, and integer/boolean conventions before reconstructing JAX
arrays. Corrupt or incompatible entries rebuild under `auto`; `readonly` rejects
them without writing.

## Fresh-process TPG results

| Case | Assignment (s) | Problem prep. (s) | Compile load (s) | Warm V+G (s) | Optimizer (s) | Process (s) |
|---|---:|---:|---:|---:|---:|---:|
| Cache off | 0.0971 | 1.1756 | 0.0067 | 0.000992 | 0.2908 | 2.1939 |
| Empty cache/populate | 0.3412 | 1.3532 | 0.0070 | 0.000993 | 0.2975 | 2.4106 |
| Valid hit | 0.1084 | 1.0775 | 0.0065 | 0.000937 | 0.2721 | 2.0211 |
| Truncated entry/rebuild | 0.3201 | 1.3339 | 0.0087 | 0.000965 | 0.3073 | 2.4081 |
| Provenance mismatch/rebuild | 0.3321 | 1.3463 | 0.006--0.009 | 0.001048 | 0.3166 | 2.3935 |
| Read-only hit | 0.1128 | 1.1242 | 0.0068 | 0.000972 | 0.2922 | 2.1872 |

For the representative valid hit, opening the NPZ took 0.00044 s,
decompression 0.00460 s, validation (including decompression and structural
checks) 0.01473 s, host reconstruction 0.000006 s, and JAX transfer plus explicit
synchronization 0.00500 s. Canonical provenance generation took 0.08787 s and is
now the dominant assignment-cache operation.

After the grouping fix, rebuilding is already roughly as fast as strict loading;
the cache is therefore most valuable as a reproducible validated artifact, not
as a large incremental speedup over the simplified builder. It nevertheless
reduces assignment loading by more than 30 times relative to the original
3.313 s baseline, and complete one-iteration time falls from 5.389 s to about
2.0--2.2 s. Avoiding the strong fingerprint would make cache loading appear
faster but would weaken invalidation and was deliberately rejected.

Valid cache-hit runs produced objectives 55,243.4375, 49,635.4922, and
49,140.4258 at 1, 5, and 20 iterations, exactly matching the preceding cached
operator benchmark. Their complete process times were 2.021, 4.413, and 7.276 s.
Integer, boolean, ordering, grouping, sentinel, and fingerprint fields are tested
with exact equality; floating flows, predictions, objectives, and gradients use
the existing float32 tolerances.

Raw results are `tpg_assignment_cache*.json`; the reusable archive is in
`tpg_assignment_cache/`.
