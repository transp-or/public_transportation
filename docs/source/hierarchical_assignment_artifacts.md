# Hierarchical scheduled-assignment artifacts

Scheduled temporal assignment is persisted as a parent-aware DAG.  The
final layer, `estimation_assignment_mapping` (L7), is the exact linear mapping
consumed by estimation:

```text
predicted_measurements = fixed_offset + A_free @ d_free
```

The layers are:

```text
L0 scenario_base
L1 canonical_od_time_universe
L2 feasibility_support
L3 route_choice_basis
L4 route_choice_materialization
L5 full_od_assignment_mapping
L6 observation_projection
L7 estimation_assignment_mapping
```

The nodes are topologically ordered, but manifests record only their direct
scientific parents:

```text
scenario_base
└── canonical_od_time_universe
    └── feasibility_support
        └── route_choice_basis
            └── route_choice_materialization(theta, dtype)
                └── full_od_assignment_mapping
                        ├── observation_projection(measurement mapping)
                        └──────────────┐
                                       ▼
                        estimation_assignment_mapping
```

L7 is a join node: it depends directly on both the full assignment mapping and
the observation projection. Future observation branches can therefore reuse
the same L5 mapping without invalidating it.

Each manifest records its layer, schema version, fingerprint, direct parent
fingerprints, scientific input fingerprints, execution metadata, payload
checksums, and completion status.  The fingerprint is computed from canonical
JSON containing only the artifact type, schema version, parent fingerprints,
scientific parameters, and input fingerprints.  Worker count, batch size,
temporary directories, deadlines, and progress intervals are execution
parameters and never change scientific identity.

The DAG layout is schema version 3. Artifacts from the earlier linear layout
are incompatible and must be regenerated explicitly.

## Reuse policy

Preparation defaults to `checkpoint_policy="reuse_or_build"`.  It first takes
a metadata-only fast path: it checks the manifest, package/configuration and
parent identities, the `COMPLETE.json` marker, declared output files, and
recorded file sizes.  It reuses a complete match, resumes a matching
incomplete checkpoint, builds a missing artifact, and fails rather than
silently rebuilding an incompatible artifact.  Use `verify="full"` only for
an explicit integrity audit; the normal reuse path does not hash every
payload.

`checkpoint_policy="reuse_only"` is the fail-closed policy for fit,
validation, reporting, identifiability, and export.  `checkpoint_policy="rebuild"`
is the explicit invalidation request.  The historical
`activation_policy="build_or_reuse"` and `"force_rebuild"` names remain
accepted by the preparation API as compatibility aliases.

Every published artifact has a manifest and a `COMPLETE.json` marker.  The
marker is written only after payload publication and binds the identity to the
manifest hash.  A missing marker, missing output, size mismatch, or incompatible
metadata produces a structured `HierarchicalArtifactUnavailableError` with a
stable reason code; an artifact from the obsolete monolithic format raises
`ObsoleteArtifactFormatError`.

An explicit preparation command may use the new policies:

```python
checkpoint_policy="reuse_or_build"
checkpoint_policy="rebuild"
```

Fit, validation, reporting, identifiability, and viewer/export workflows must
always use `reuse_only`.  They must not invoke routing, assignment
construction, or an implicit fallback builder.

Checkpoints and completed artifacts have separate namespaces:

```text
<results>/checkpoints/<layer>/<fingerprint>/
<results>/artifacts/<layer>/<fingerprint>/
```

Partial directories are never accepted as completed layers.

Gravity run manifests record the L7 fingerprint, its direct DAG parent
fingerprints, the mapping shape, free-OD and observation counts, and the
fixed-offset fingerprint.
Validation rejects a fit result when the problem supplies a different L7
artifact, so downstream diagnostics cannot silently mix operators.

## Invalidation rules

The hierarchy permits downstream changes without repeating unrelated work:

```text
theta change:
    rebuild L4-L7 only

fixed-positive demand value change:
    update the L7 fixed offset only

structural-zero mask change:
    rebuild L7 only

measurement-value change:
    no operator rebuild

measurement mapping change:
    rebuild L6-L7

likelihood or gravity-model change:
    no routing-artifact rebuild

feasibility-rule change:
    rebuild L2 and all descendants

timetable or network change:
    rebuild all descendants
```

The physical OD universe at L1 retains structural-zero cells.  L5 is defined
over that full canonical order.  L7 persists the compact free-column mapping,
full-to-compact and compact-to-full indices, measurement-row ordering,
fixed-positive offsets, and the fingerprints needed to verify those mappings.

## Progress

`HierarchicalProgressReporter` writes both a per-run JSONL stream and an
optional campaign-level JSONL stream.  Events contain the layer and artifact
fingerprints, activation policy, status, current unit, completed and total
work, elapsed time, ETA, confidence, and whether the layer was reused or
rebuilt.  Valid statuses are `started`, `running`, `heartbeat`, `reused`,
`completed`, `failed`, and `interrupted`.  If total work is unknown, ETA is
`null` and confidence is `unavailable`.

## Minimal API

```python
from public_transportation.inference import (
    EstimationAssignmentMapping,
    HierarchicalProgressReporter,
    prepare_hierarchical_assignment_mapping,
)

prepared = prepare_hierarchical_assignment_mapping(
    root=results_root,
    specifications=layer_specifications,
    mapping_builder=build_l7_mapping,
    checkpoint_policy="reuse_or_build",
    verify="fast",
    progress=HierarchicalProgressReporter(
        results_root / "progress.jsonl",
        campaign_path=results_root / "campaign_progress.jsonl",
    ),
)

# Later processes use only the persisted final mapping.
reused = prepare_hierarchical_assignment_mapping(
    root=results_root,
    specifications=layer_specifications,
    checkpoint_policy="reuse_only",
    verify="fast",
)
mapping = reused.mapping
prediction = mapping.fixed_offset + mapping.matvec(d_free)
```

For a matrix-free companion backend, `EstimationAssignmentMapping.to_payload`
records `matrix_format="companion_operator"`; loading it requires an explicit
`companion_loader` callback.  This prevents an estimator from silently
substituting a different numerical operator.

The case-template driver exposes the full audit as
`uv run --frozen python run_case.py prepare --verify`; the API equivalent is
`verify="full"`.

The existing scheduled operator uses the same L7 contract without allocating a
dense full-network matrix.  Its sparse/block backend is the numerical payload;
the hierarchy manifest remains the authoritative provenance and reuse gate.
Preparation and activation accept an optional `HierarchicalProgressReporter`
(`hierarchical_progress=`).  Pass one with both a per-run path and
`campaign_path=` to receive the same layer events in two durable JSONL streams;
the existing construction reporter continues to carry low-level shard timing.

## Campaign workflow

1. Prepare all layers with `checkpoint_policy="reuse_or_build"`; matching
   checkpoints resume automatically and complete artifacts are reused.
2. Use `verify="full"` for an occasional integrity audit, not for every
   activation.
3. Run fit, validation, identifiability, reporting, and viewer/export with
   `checkpoint_policy="reuse_only"`.
4. Rerun those downstream stages to confirm that all layers emit `reused`
   events and no construction event occurs.

Do not convert, reinterpret, or silently reuse artifacts created by the old
monolithic direct-temporal format.  Regenerate the hierarchy instead.
