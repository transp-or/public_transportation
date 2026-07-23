# Simple Example 02

Synthetic route-choice bus network. The reference data are in `data/`.

`data/fixed_demand.csv` is the sparse list of OD/time-bin cells excluded from
estimation. A blank or omitted `fixed_flow` means zero; an explicit value fixes
the cell at that nonnegative flow. All estimation scripts read this file and
use the same reduced layout for Bayesian VI, maximum likelihood, and MAP.
