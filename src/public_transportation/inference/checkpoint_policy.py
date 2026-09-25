"""Common policies for persistent scientific artifacts and checkpoints."""

from __future__ import annotations

from typing import Literal


CheckpointPolicy = Literal["reuse_or_build", "reuse_only", "rebuild"]


def normalize_checkpoint_policy(
    policy: str | None = None,
    *,
    legacy_policy: str | None = None,
    resume: bool | None = None,
) -> CheckpointPolicy:
    """Normalize the new policy and the historical boolean/activation names.

    ``resume`` remains accepted for source compatibility, but callers should
    use ``policy``.  A false legacy flag no longer disables automatic reuse;
    an explicit ``rebuild`` policy is required for that behavior.
    """

    selected = policy if policy is not None else legacy_policy
    aliases = {
        "build_or_reuse": "reuse_or_build",
        "force_rebuild": "rebuild",
    }
    if selected is None:
        selected = "reuse_or_build"
    selected = aliases.get(str(selected), str(selected))
    if resume is True and policy is None:
        selected = "reuse_or_build"
    if selected not in {"reuse_or_build", "reuse_only", "rebuild"}:
        raise ValueError(
            "checkpoint policy must be 'reuse_or_build', 'reuse_only', or 'rebuild'"
        )
    return selected  # type: ignore[return-value]


__all__ = ["CheckpointPolicy", "normalize_checkpoint_policy"]
