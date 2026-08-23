"""Optional aggregate-observation channels for gravity estimation.

This module defines the extension point used by later gravity-observation
implementations (for example, aggregate GPS trip-attribute distributions).
The phase-1 interface is intentionally independent of any particular file
format or likelihood.  A channel owns its observed aggregate, its prediction
from free OD demand, and its JAX-compatible likelihood contribution.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import jax
import numpy as np

from public_transportation.inference.block_coordinate._canonical import (
    canonical_json,
    fingerprint,
)


@runtime_checkable
class GravityObservationChannel(Protocol):
    """Protocol implemented by one optional aggregate observation channel.

    ``fingerprint`` must identify the observed aggregate and all semantic
    settings that affect its prediction or likelihood.  ``report`` is for
    human- and machine-readable stage summaries; it is deliberately kept
    separate from the identity fingerprint.
    """

    @property
    def name(self) -> str: ...

    @property
    def kind(self) -> str: ...

    @property
    def observed(self) -> object: ...

    @property
    def fingerprint(self) -> str: ...

    def validate(self, *, num_free_od: int, dtype: np.dtype) -> None: ...

    def predict(self, demand: jax.Array) -> jax.Array: ...

    def log_likelihood(
        self, *, prediction: jax.Array, raw_parameters: jax.Array
    ) -> jax.Array: ...

    def report(self) -> Mapping[str, object]: ...


@dataclass(frozen=True, slots=True)
class GravityObservationBundle:
    """Immutable collection of optional aggregate-observation channels.

    The empty bundle is the default and has no effect on the existing
    count-only objective.  A non-empty bundle contributes an identity payload
    to the gravity model fingerprint, while each channel remains responsible
    for its eventual prediction and likelihood implementation.
    """

    channels: tuple[GravityObservationChannel, ...] = ()

    def __post_init__(self) -> None:
        channels = tuple(self.channels)
        object.__setattr__(self, "channels", channels)
        names: set[str] = set()
        for channel in channels:
            if not isinstance(channel, GravityObservationChannel):
                raise TypeError(
                    "observation channels must implement GravityObservationChannel."
                )
            name = channel.name
            if not isinstance(name, str) or not name.strip():
                raise ValueError("observation channel name must be a non-empty string.")
            if name in names:
                raise ValueError(f"observation channel names must be unique: {name!r}.")
            names.add(name)
            kind = channel.kind
            if not isinstance(kind, str) or not kind.strip():
                raise ValueError(
                    f"observation channel {name!r} kind must be a non-empty string."
                )
            channel_fingerprint = channel.fingerprint
            if (
                not isinstance(channel_fingerprint, str)
                or not channel_fingerprint.strip()
            ):
                raise ValueError(
                    f"observation channel {name!r} fingerprint must be non-empty."
                )
            report = channel.report()
            if not isinstance(report, Mapping):
                raise TypeError(
                    f"observation channel {name!r} report must be a mapping."
                )
            try:
                canonical_json(dict(report))
            except (TypeError, ValueError) as error:
                raise TypeError(
                    f"observation channel {name!r} report must be serializable."
                ) from error

    @classmethod
    def empty(cls) -> GravityObservationBundle:
        """Return the canonical disabled bundle."""

        return cls()

    @property
    def enabled(self) -> bool:
        """Whether at least one optional channel is enabled."""

        return bool(self.channels)

    @property
    def fingerprint(self) -> str | None:
        """Stable identity for enabled channels, or ``None`` when empty."""

        if not self.enabled:
            return None
        return fingerprint(self.identity_payload())

    def identity_payload(self) -> dict[str, object]:
        """Return only semantic identity fields, excluding report diagnostics."""

        return {
            "schema_version": 1,
            "channels": [
                {
                    "name": channel.name,
                    "kind": channel.kind,
                    "fingerprint": channel.fingerprint,
                }
                for channel in self.channels
            ],
        }

    def to_dict(self) -> dict[str, object]:
        """Return a serializable channel/report breakdown for manifests."""

        return {
            **self.identity_payload(),
            "channels": [
                {
                    "name": channel.name,
                    "kind": channel.kind,
                    "fingerprint": channel.fingerprint,
                    "report": dict(channel.report()),
                }
                for channel in self.channels
            ],
        }

    def validate(self, *, num_free_od: int, dtype: np.dtype) -> None:
        """Validate every enabled channel against the current OD dimension."""

        if num_free_od <= 0:
            raise ValueError("num_free_od must be positive for observation validation.")
        for channel in self.channels:
            channel.validate(num_free_od=num_free_od, dtype=dtype)

    def reports(self) -> tuple[Mapping[str, object], ...]:
        """Return the channel report breakdown in deterministic channel order."""

        return tuple(channel.report() for channel in self.channels)


__all__ = ["GravityObservationBundle", "GravityObservationChannel"]
