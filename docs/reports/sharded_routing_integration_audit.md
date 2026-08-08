# Fixed-routing sharding integration audit

This note records the public data flow before the direct scheduled builder was
adapted to persistent destination-batched routing. It intentionally contains no
private scenario data.

## Existing data flow

`AssignmentInputs` owns the graph, base link costs, destination-group nodes and
link masks, and compact OD/group indexing. `_prepare_fixed_routing_core` scans
the complete leading destination-group dimension and returns two dense arrays:
the effective mask and link probability for every group and link.
`prepare_fixed_routing` traces, lowers, compiles, executes, synchronizes and
transfers that complete operation before constructing `FixedRoutingInputs`.

The direct scheduled activation path then serializes those two complete arrays
to `routing.npz`. Consequently, its first durable routing boundary follows the
all-group synchronization and host transfer. Origin-support discovery converts
both arrays to NumPy and indexes them by group. The sharded measurement builder
does the same during support planning and indexes the dense JAX arrays again
during numerical shard construction. Its construction plan stores group/task
identities rather than probabilities, but the complete routing object remains
resident throughout construction. Validation compares its complete source
mask, destination ordering, graph dimensions and base costs with
`AssignmentInputs`. Construction provenance is derived from assignment inputs,
mapping, layout, theta and configuration; the old direct `routing.npz` has its
own content hash but no internal completion boundary.

## Dense assumptions to remove from the direct path

1. `FixedRoutingInputs.effective_group_link_mask` and
   `group_link_probability` are assumed to have shape `(groups, links)`.
2. `validate_fixed_routing_compatibility` requires those complete shapes.
3. Origin-support discovery calls `np.asarray` on both complete arrays before
   its group loop.
4. Sharded support planning calls `np.asarray` on the complete effective mask.
5. Numerical construction indexes the complete routing arrays for each task.
6. The direct builder persists and reloads a single dense `routing.npz` and
   calls its routing factory before any support checkpoint can be created.

## Existing reusable sharded implementation

`sharded_fixed_routing.py` already provides deterministic contiguous shard
planning, fixed-shape padded execution, compilation reuse, atomic per-shard NPZ
files, a resumable manifest, strict scientific identity, resource ceilings,
deadline guards, diagnostics, and `ShardedFixedRoutingInputs`. It deliberately
retains no global probability or effective-mask array. This implementation is
already tested independently, but the direct scheduled builder and its support
and numerical construction path do not consume it.

The implementation work therefore integrates this existing public sharded
contract through a common dense/sharded adapter. It does not create a second
routing format and does not concatenate routing shards. The dense
`FixedRoutingInputs` path remains available for small examples and backward
compatibility.
