from collections.abc import Callable
from typing import Any, Literal

from numpyro.infer.autoguide import (
    AutoDiagonalNormal,
    AutoLowRankMultivariateNormal,
    AutoMultivariateNormal,
    AutoNormal,
)


def make_autoguide(
    *,
    model: Callable[[], None],
    guide: Literal["auto_diag", "auto_lowrank", "auto_mvn", "auto_normal"] = "auto_diag",
    lowrank_rank: int | None = None,
) -> Any:
    """
    Create a NumPyro autoguide for the given model.

    :param model: NumPyro model callable.
    :param guide: Choice of autoguide:
        - "auto_diag": AutoDiagonalNormal (mean-field Gaussian). Best default for very high d.
        - "auto_lowrank": Low-rank + diagonal Gaussian. Captures correlations at moderate cost.
        - "auto_mvn": Full-covariance Gaussian. Usually infeasible for thousands of parameters.
        - "auto_normal": AutoNormal (kept for completeness; often similar spirit).
    :param lowrank_rank: Rank for "auto_lowrank". If None, a conservative default is used.
    :return: A NumPyro autoguide instance.
    """
    if guide == "auto_diag":
        return AutoDiagonalNormal(model)
    if guide == "auto_lowrank":
        rank = 20 if lowrank_rank is None else int(lowrank_rank)
        if rank <= 0:
            raise ValueError("lowrank_rank must be a positive integer.")
        return AutoLowRankMultivariateNormal(model, rank=rank)
    if guide == "auto_mvn":
        return AutoMultivariateNormal(model)
    if guide == "auto_normal":
        return AutoNormal(model)
    raise ValueError(f"Unknown guide: {guide!r}")

