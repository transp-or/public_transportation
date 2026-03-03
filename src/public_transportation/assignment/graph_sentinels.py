"""
Centralized sentinel constants for the time-expanded graph.

Why this exists
---------------
We use *finite* sentinel time values (instead of ±inf) to place centroid nodes at
conceptual -∞ and +∞ in the graph's topological ordering.

These sentinels must be defined in a single location so that:
- the builder(s) and report/debug scripts agree on the exact values,
- we avoid scattered magic numbers (e.g., -1e12, +1e12) across modules.

Design notes
------------
- We keep these as implementation constants, not as user configuration.
  (Centroid-in nodes are never time-tagged in the design.)
- Values are in MINUTES for node_time_min (consistent with JaxGraph.node_time).
- node_time_s uses -1 for centroids (seconds-from-midnight is meaningful only for event nodes).

If you later want the graph to be self-describing, also store these values inside
the JaxGraph dataclass (e.g., sentinel_time_min_centroid_in/out).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Node-time sentinels (minutes)
# ---------------------------------------------------------------------------

# Large finite values used as surrogates for conceptual -∞ and +∞.
# Must be far outside the range of any real event times in minutes.
#
# With seconds-from-midnight converted to minutes, typical event times are in [0, 1440).
# Using 1e12 minutes is safely separated, while remaining finite and stable in JAX.
CENTROID_IN_TIME_MIN: float = -1.0e12
CENTROID_OUT_TIME_MIN: float = +1.0e12

# ---------------------------------------------------------------------------
# node_time_s sentinel for centroids (seconds-from-midnight)
# ---------------------------------------------------------------------------

# For event nodes, node_time_s stores seconds-from-midnight (int).
# For centroid nodes (in/out), no physical time exists; we store -1.
CENTROID_TIME_S: int = -1

# ---------------------------------------------------------------------------
# Node-kind codes (optional: centralize if multiple modules rely on them)
# ---------------------------------------------------------------------------

NODE_KIND_CENTROID_IN: int = 0
NODE_KIND_EVENT_ARR: int = 1
NODE_KIND_EVENT_DEP: int = 2
NODE_KIND_CENTROID_OUT: int = 3

# ---------------------------------------------------------------------------
# Link-type codes (optional: centralize if multiple modules rely on them)
# ---------------------------------------------------------------------------

LINK_TYPE_RIDE: int = 0
LINK_TYPE_TRANSFER: int = 1
LINK_TYPE_ACCESS: int = 2
LINK_TYPE_EGRESS: int = 3
LINK_TYPE_DWELL: int = 4