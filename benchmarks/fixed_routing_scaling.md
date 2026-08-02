# Fixed-routing scaling and parallel preparation

Controlled DAG experiments varied nodes, links, groups, maximum outgoing degree,
and enabled-link density. For a fixed graph and group count, changing enabled
density from 10% to 100% had essentially no effect on warm execution. Increasing
groups, nodes, links, or degree increased runtime. This agrees with code
inspection: every destination evaluates the reverse dynamic program over all
nodes and computes probabilities through all links. The complete-link kernel is
therefore approximately `O(groups * (nodes * max_out_degree + links))`, not
`O(groups * enabled_support)`.

The shared-executable thread benchmark used 16 groups on an 8,192-node synthetic
DAG, divided into eight shards. Serial, two-worker, and four-worker results were
numerically identical and each configuration compiled one executable. The
recorded run measured 0.3180, 0.0801, and 0.0446 seconds respectively. These
small-case timings include cache and runtime warm-up effects and are descriptive,
not portable speed guarantees. CI asserts structure and numerical identity only.

Threads were selected because they share the multi-gigabyte assignment graph and
compiled executable; spawned JAX processes would duplicate both. XLA releases the
Python GIL during execution. Global RSS admission limits concurrent shards, and
the scheduler reserves a safety margin before every dispatch.

Support-restricted routing is technically justified by the complexity result but
is not enabled yet. The packaged Simple Example 02 has approximately 91% effective
link density, so compact support would not help it. The new synchronized shard
diagnostics must first measure full-network effective density. A sparse-support
schema should be introduced only if that measurement shows enough reduction to
offset compact-index preparation and scatter costs.
