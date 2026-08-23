"""Versioned, privacy-safe aggregate observations for gravity estimation.

The Phase-2 contract deliberately contains only aggregate histograms and
metadata.  It does not contain individual GPS traces and it does not construct
the route-response operator or a likelihood.  A document has the following
shape (the checksum is the SHA-256 fingerprint of the canonical document with
the ``content_sha256`` field omitted)::

    {
      "schema_version": 1,
      "channel_name": "gps_trip_attributes",
      "kind": "trip_attribute_distribution",
      "histograms": [{
        "attribute": "travel_time",
        "unit": "seconds",
        "support": [0, 7200],
        "bins": [
          {"label": "0_600", "lower": 0, "upper": 600},
          {"label": "600_7200", "lower": 600, "upper": 7200}
        ],
        "strata": [{"name": "all", "counts": [10, 20], "total": 30}]
      }],
      "metadata": {
        "collection_period": "2022-10-03/2022-12-23",
        "valid_journeys": 30,
        "excluded_journeys": 0,
        "cleaning_reasons": {},
        "apc_overlap_policy": "overlap_recorded_not_independent"
      },
      "uncertainty": {"likelihood": "dirichlet_multinomial", "concentration": 50},
      "content_sha256": "..."
    }

The contract is intentionally strict.  A later schema version can add richer
joint attributes without making a malformed version-1 file appear valid.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from public_transportation.inference.block_coordinate._canonical import fingerprint

GRAVITY_AGGREGATE_SCHEMA_VERSION = 1

SUPPORTED_AGGREGATE_LIKELIHOODS = frozenset(
    {"multinomial", "dirichlet_multinomial", "tempered_multinomial"}
)
SUPPORTED_AGGREGATE_UNITS = frozenset({"seconds", "metres", "meters", "stops", "count"})

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "channel_name",
        "kind",
        "histograms",
        "metadata",
        "uncertainty",
        "content_sha256",
    }
)
_HISTOGRAM_KEYS = frozenset({"attribute", "unit", "support", "bins", "strata"})
_BIN_KEYS = frozenset({"label", "lower", "upper"})
_STRATUM_KEYS = frozenset({"name", "counts", "total"})
_METADATA_KEYS = frozenset(
    {
        "collection_period",
        "valid_journeys",
        "excluded_journeys",
        "cleaning_reasons",
        "apc_overlap_policy",
        "strata_exhaustive",
    }
)
_UNCERTAINTY_KEYS = frozenset(
    {"likelihood", "concentration", "effective_sample_size", "tempering"}
)


def _mapping(value: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object.")
    return value


def _reject_unknown(
    payload: Mapping[str, object], allowed: frozenset[str], *, context: str
) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unsupported {context} field(s): {', '.join(unknown)}.")


def _string(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string.")
    return value.strip()


def _finite(value: object, *, context: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{context} must be finite.")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} must be finite.") from error
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite.")
    return result


def _integer(value: object, *, context: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{context} must be a non-negative integer.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} must be a non-negative integer.") from error
    if not math.isfinite(numeric) or not numeric.is_integer() or numeric < 0:
        raise ValueError(f"{context} must be a non-negative integer.")
    return int(numeric)


def _same_boundary(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1.0e-12, abs_tol=1.0e-12)


@dataclass(frozen=True, slots=True)
class GravityAggregateBin:
    """One closed-open numeric bin, ``[lower, upper)``."""

    label: str
    lower: float
    upper: float

    def to_dict(self) -> dict[str, object]:
        return {"label": self.label, "lower": self.lower, "upper": self.upper}


@dataclass(frozen=True, slots=True)
class GravityAggregateStratum:
    """Observed histogram counts for one declared stratum."""

    name: str
    counts: tuple[int, ...]
    total: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("aggregate stratum name must not be empty.")
        if self.total < 0 or sum(self.counts) != self.total:
            raise ValueError(
                f"aggregate stratum {self.name!r} total must equal the sum of counts."
            )

    def to_dict(self, labels: tuple[str, ...]) -> dict[str, object]:
        return {
            "name": self.name,
            "counts": dict(zip(labels, self.counts, strict=True)),
            "total": self.total,
        }


@dataclass(frozen=True, slots=True)
class GravityAggregateHistogram:
    """One attribute histogram, optionally split into strata."""

    attribute: str
    unit: str
    support: tuple[float, float]
    bins: tuple[GravityAggregateBin, ...]
    strata: tuple[GravityAggregateStratum, ...]

    def __post_init__(self) -> None:
        if not self.attribute:
            raise ValueError("aggregate histogram attribute must not be empty.")
        if self.unit not in SUPPORTED_AGGREGATE_UNITS:
            raise ValueError(f"unsupported aggregate unit: {self.unit!r}.")
        if len(self.support) != 2 or self.support[0] >= self.support[1]:
            raise ValueError("aggregate histogram support must be an increasing pair.")
        if not self.bins:
            raise ValueError("aggregate histogram must define at least one bin.")
        if not self.strata:
            raise ValueError("aggregate histogram must define at least one stratum.")

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(item.label for item in self.bins)

    @property
    def sample_size(self) -> int:
        return sum(item.total for item in self.strata)

    def to_dict(self) -> dict[str, object]:
        labels = self.labels
        return {
            "attribute": self.attribute,
            "unit": self.unit,
            "support": list(self.support),
            "bins": [item.to_dict() for item in self.bins],
            "strata": [item.to_dict(labels) for item in self.strata],
        }


@dataclass(frozen=True, slots=True)
class GravityAggregateUncertainty:
    """Declared uncertainty treatment for the aggregate sample."""

    likelihood: str
    concentration: float | None = None
    effective_sample_size: float | None = None
    tempering: float | None = None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"likelihood": self.likelihood}
        if self.concentration is not None:
            result["concentration"] = self.concentration
        if self.effective_sample_size is not None:
            result["effective_sample_size"] = self.effective_sample_size
        if self.tempering is not None:
            result["tempering"] = self.tempering
        return result


@dataclass(frozen=True, slots=True)
class GravityAggregateObservation:
    """Validated aggregate data and provenance returned by the loader."""

    schema_version: int
    channel_name: str
    kind: str
    histograms: tuple[GravityAggregateHistogram, ...]
    metadata: Mapping[str, object]
    uncertainty: GravityAggregateUncertainty
    source_path: Path
    file_sha256: str
    content_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != GRAVITY_AGGREGATE_SCHEMA_VERSION:
            raise ValueError("unsupported aggregate observation schema version.")
        if not self.histograms:
            raise ValueError("aggregate observation must contain a histogram.")
        names = tuple(item.attribute for item in self.histograms)
        if len(set(names)) != len(names):
            raise ValueError("aggregate histogram attributes must be unique.")
        object.__setattr__(self, "source_path", Path(self.source_path).resolve())
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def fingerprint(self) -> str:
        """Semantic identity independent of the local source path."""

        return fingerprint(self.identity_payload())

    @property
    def valid_journeys(self) -> int:
        return int(self.metadata["valid_journeys"])

    @property
    def excluded_journeys(self) -> int:
        return int(self.metadata["excluded_journeys"])

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "channel_name": self.channel_name,
            "kind": self.kind,
            "content_sha256": self.content_sha256,
            "uncertainty": self.uncertainty.to_dict(),
        }

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "channel_name": self.channel_name,
            "kind": self.kind,
            "histograms": [item.to_dict() for item in self.histograms],
            "metadata": dict(self.metadata),
            "uncertainty": self.uncertainty.to_dict(),
        }
        payload["content_sha256"] = fingerprint(payload)
        return payload

    def audit_payload(self, *, status: str = "validated") -> dict[str, object]:
        return {
            "schema_version": 1,
            "status": status,
            "source_path": str(self.source_path),
            "file_sha256": self.file_sha256,
            "content_sha256": self.content_sha256,
            "aggregate_fingerprint": self.fingerprint,
            "channel_name": self.channel_name,
            "kind": self.kind,
            "valid_journeys": self.valid_journeys,
            "excluded_journeys": self.excluded_journeys,
            "histograms": [
                {
                    "attribute": item.attribute,
                    "unit": item.unit,
                    "support": list(item.support),
                    "bin_count": len(item.bins),
                    "bins": [bin_item.to_dict() for bin_item in item.bins],
                    "strata": [
                        {"name": stratum.name, "total": stratum.total}
                        for stratum in item.strata
                    ],
                }
                for item in self.histograms
            ],
            "uncertainty": self.uncertainty.to_dict(),
            "metadata": dict(self.metadata),
        }


def _parse_bins(
    value: object, *, support: tuple[float, float], context: str
) -> tuple[GravityAggregateBin, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{context}.bins must be an array.")
    parsed: list[GravityAggregateBin] = []
    labels: set[str] = set()
    for index, raw in enumerate(value):
        payload = _mapping(raw, context=f"{context}.bins[{index}]")
        _reject_unknown(payload, _BIN_KEYS, context=f"{context}.bins[{index}]")
        label = _string(payload.get("label"), context=f"{context}.bins[{index}].label")
        if label in labels:
            raise ValueError(f"duplicate bin label {label!r} in {context}.")
        labels.add(label)
        lower = _finite(payload.get("lower"), context=f"{context}.bins[{index}].lower")
        upper = _finite(payload.get("upper"), context=f"{context}.bins[{index}].upper")
        if lower >= upper:
            raise ValueError(f"{context}.bins[{index}] must have lower < upper.")
        if lower < support[0] or upper > support[1]:
            raise ValueError(f"{context}.bins[{index}] lies outside the support.")
        parsed.append(GravityAggregateBin(label, lower, upper))
    parsed.sort(key=lambda item: (item.lower, item.upper, item.label))
    if not parsed:
        raise ValueError(f"{context}.bins must not be empty.")
    if not _same_boundary(parsed[0].lower, support[0]):
        raise ValueError(f"{context}.bins do not cover the lower support boundary.")
    previous = parsed[0]
    for current in parsed[1:]:
        if current.lower < previous.upper and not _same_boundary(
            current.lower, previous.upper
        ):
            raise ValueError(f"{context}.bins overlap.")
        if not _same_boundary(current.lower, previous.upper):
            raise ValueError(f"{context}.bins contain a gap and are incomplete.")
        previous = current
    if not _same_boundary(parsed[-1].upper, support[1]):
        raise ValueError(f"{context}.bins do not cover the upper support boundary.")
    return tuple(parsed)


def _parse_strata(
    value: object, *, labels: tuple[str, ...], context: str
) -> tuple[GravityAggregateStratum, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{context}.strata must be an array.")
    parsed: list[GravityAggregateStratum] = []
    names: set[str] = set()
    for index, raw in enumerate(value):
        payload = _mapping(raw, context=f"{context}.strata[{index}]")
        _reject_unknown(payload, _STRATUM_KEYS, context=f"{context}.strata[{index}]")
        name = _string(payload.get("name"), context=f"{context}.strata[{index}].name")
        if name in names:
            raise ValueError(f"duplicate stratum name {name!r} in {context}.")
        names.add(name)
        raw_counts = payload.get("counts")
        if isinstance(raw_counts, Mapping):
            if set(raw_counts) != set(labels):
                raise ValueError(
                    f"{context}.strata[{index}].counts must name every bin exactly once."
                )
            counts = tuple(
                _integer(
                    raw_counts[label],
                    context=f"{context}.strata[{index}].counts[{label!r}]",
                )
                for label in labels
            )
        elif isinstance(raw_counts, Sequence) and not isinstance(
            raw_counts, (str, bytes)
        ):
            if len(raw_counts) != len(labels):
                raise ValueError(
                    f"{context}.strata[{index}].counts length must equal the bin count."
                )
            counts = tuple(
                _integer(item, context=f"{context}.strata[{index}].counts[{position}]")
                for position, item in enumerate(raw_counts)
            )
        else:
            raise ValueError(
                f"{context}.strata[{index}].counts must be an array or object."
            )
        total = _integer(
            payload.get("total"), context=f"{context}.strata[{index}].total"
        )
        if sum(counts) != total:
            raise ValueError(
                f"{context}.strata[{index}].total must equal the sum of counts."
            )
        parsed.append(GravityAggregateStratum(name, counts, total))
    if not parsed:
        raise ValueError(f"{context}.strata must not be empty.")
    return tuple(parsed)


def _parse_histograms(value: object) -> tuple[GravityAggregateHistogram, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("histograms must be an array.")
    parsed: list[GravityAggregateHistogram] = []
    attributes: set[str] = set()
    for index, raw in enumerate(value):
        context = f"histograms[{index}]"
        payload = _mapping(raw, context=context)
        _reject_unknown(payload, _HISTOGRAM_KEYS, context=context)
        attribute = _string(payload.get("attribute"), context=f"{context}.attribute")
        if attribute in attributes:
            raise ValueError(f"duplicate histogram attribute {attribute!r}.")
        attributes.add(attribute)
        unit = _string(payload.get("unit"), context=f"{context}.unit")
        if unit not in SUPPORTED_AGGREGATE_UNITS:
            raise ValueError(f"unsupported aggregate unit {unit!r} for {attribute!r}.")
        raw_support = payload.get("support")
        if not isinstance(raw_support, Sequence) or len(raw_support) != 2:
            raise ValueError(f"{context}.support must be an increasing pair.")
        support = (
            _finite(raw_support[0], context=f"{context}.support[0]"),
            _finite(raw_support[1], context=f"{context}.support[1]"),
        )
        if support[0] >= support[1]:
            raise ValueError(f"{context}.support must be increasing.")
        bins = _parse_bins(payload.get("bins"), support=support, context=context)
        strata = _parse_strata(
            payload.get("strata"),
            labels=tuple(item.label for item in bins),
            context=context,
        )
        parsed.append(GravityAggregateHistogram(attribute, unit, support, bins, strata))
    if not parsed:
        raise ValueError("histograms must not be empty.")
    return tuple(parsed)


def _parse_metadata(value: object) -> dict[str, object]:
    payload = _mapping(value, context="metadata")
    _reject_unknown(payload, _METADATA_KEYS, context="metadata")
    required = (
        "collection_period",
        "valid_journeys",
        "excluded_journeys",
        "cleaning_reasons",
        "apc_overlap_policy",
    )
    for key in required:
        if key not in payload:
            raise ValueError(f"metadata.{key} is required.")
    result = dict(payload)
    result["collection_period"] = _string(
        payload["collection_period"], context="metadata.collection_period"
    )
    result["valid_journeys"] = _integer(
        payload["valid_journeys"], context="metadata.valid_journeys"
    )
    result["excluded_journeys"] = _integer(
        payload["excluded_journeys"], context="metadata.excluded_journeys"
    )
    result["apc_overlap_policy"] = _string(
        payload["apc_overlap_policy"], context="metadata.apc_overlap_policy"
    )
    if not isinstance(payload["cleaning_reasons"], Mapping):
        raise ValueError("metadata.cleaning_reasons must be an object.")
    result["cleaning_reasons"] = dict(payload["cleaning_reasons"])
    strata_exhaustive = payload.get("strata_exhaustive", False)
    if not isinstance(strata_exhaustive, bool):
        raise ValueError("metadata.strata_exhaustive must be boolean.")
    result["strata_exhaustive"] = strata_exhaustive
    return result


def _parse_uncertainty(value: object) -> GravityAggregateUncertainty:
    payload = _mapping(value, context="uncertainty")
    _reject_unknown(payload, _UNCERTAINTY_KEYS, context="uncertainty")
    likelihood = _string(payload.get("likelihood"), context="uncertainty.likelihood")
    if likelihood not in SUPPORTED_AGGREGATE_LIKELIHOODS:
        raise ValueError(f"unsupported aggregate likelihood {likelihood!r}.")
    supplied = [
        key
        for key in ("concentration", "effective_sample_size", "tempering")
        if key in payload
    ]
    if likelihood == "multinomial" and supplied:
        raise ValueError(
            "multinomial uncertainty must not declare a scaling parameter."
        )
    if likelihood == "dirichlet_multinomial":
        if (
            len(
                [
                    key
                    for key in supplied
                    if key in ("concentration", "effective_sample_size")
                ]
            )
            != 1
        ):
            raise ValueError(
                "dirichlet_multinomial requires exactly one of concentration or effective_sample_size."
            )
        if "tempering" in supplied:
            raise ValueError("dirichlet_multinomial must not also declare tempering.")
    if likelihood == "tempered_multinomial":
        if supplied != ["tempering"]:
            raise ValueError("tempered_multinomial requires tempering only.")
    concentration = (
        None
        if "concentration" not in payload
        else _finite(payload["concentration"], context="uncertainty.concentration")
    )
    effective = (
        None
        if "effective_sample_size" not in payload
        else _finite(
            payload["effective_sample_size"],
            context="uncertainty.effective_sample_size",
        )
    )
    tempering = (
        None
        if "tempering" not in payload
        else _finite(payload["tempering"], context="uncertainty.tempering")
    )
    if concentration is not None and concentration <= 0:
        raise ValueError("uncertainty.concentration must be positive.")
    if effective is not None and effective <= 0:
        raise ValueError("uncertainty.effective_sample_size must be positive.")
    if tempering is not None and not 0 <= tempering <= 1:
        raise ValueError("uncertainty.tempering must lie in [0, 1].")
    return GravityAggregateUncertainty(likelihood, concentration, effective, tempering)


def _validate_semantics(
    histograms: tuple[GravityAggregateHistogram, ...], metadata: Mapping[str, object]
) -> None:
    if bool(metadata.get("strata_exhaustive", False)):
        valid = int(metadata["valid_journeys"])
        for histogram in histograms:
            total = histogram.sample_size
            if total != valid:
                raise ValueError(
                    f"histogram {histogram.attribute!r} strata total {total} "
                    f"does not equal metadata.valid_journeys {valid}."
                )


def _write_audit(path: Path | None, payload: Mapping[str, object]) -> None:
    if path is None:
        return
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(dict(payload), sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _rejected_audit(
    path: Path, *, file_sha256: str | None, error: Exception
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "rejected",
        "source_path": str(path.resolve()),
        "file_sha256": file_sha256,
        "error_type": type(error).__name__,
        "error": str(error),
    }


def load_gravity_aggregate(
    path: str | Path | None,
    *,
    enabled: bool | None = None,
    audit_path: str | Path | None = None,
) -> GravityAggregateObservation | None:
    """Load and validate one aggregate document.

    ``enabled=False`` is the explicit no-observation path and returns ``None``
    without reading the optional file.  When ``path`` is omitted and
    ``enabled`` is not explicitly true, the optional source is also treated as
    absent.  An explicitly enabled source with no path, a missing file, or an
    invalid document raises before any expensive routing/operator work.  If an
    audit path is supplied, absent, validated, and rejected outcomes are all
    persisted there.
    """

    audit = None if audit_path is None else Path(audit_path).expanduser()
    if enabled is False or (path is None and enabled is not True):
        _write_audit(
            audit,
            {
                "schema_version": 1,
                "status": "absent",
                "reason": "optional aggregate observation is disabled",
            },
        )
        return None
    if path is None:
        error = ValueError("aggregate observation is enabled but no file was provided.")
        _write_audit(
            audit, {"schema_version": 1, "status": "rejected", "error": str(error)}
        )
        raise error
    source = Path(path).expanduser()
    if not source.is_file():
        error = FileNotFoundError(
            f"enabled aggregate observation file does not exist: {source}"
        )
        _write_audit(audit, _rejected_audit(source, file_sha256=None, error=error))
        raise error
    try:
        raw_bytes = source.read_bytes()
    except OSError as error:
        _write_audit(audit, _rejected_audit(source, file_sha256=None, error=error))
        raise
    file_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
        payload = _mapping(payload, context="aggregate observation")
        _reject_unknown(payload, _TOP_LEVEL_KEYS, context="aggregate observation")
        if int(payload.get("schema_version", -1)) != GRAVITY_AGGREGATE_SCHEMA_VERSION:
            raise ValueError(
                "unsupported aggregate observation schema_version; expected 1."
            )
        declared_checksum = _string(
            payload.get("content_sha256"), context="content_sha256"
        )
        checksum_payload = dict(payload)
        checksum_payload.pop("content_sha256", None)
        computed_checksum = fingerprint(checksum_payload)
        if declared_checksum != computed_checksum:
            raise ValueError(
                "content_sha256 does not match the canonical aggregate document."
            )
        channel_name = _string(payload.get("channel_name"), context="channel_name")
        kind = _string(payload.get("kind"), context="kind")
        histograms = _parse_histograms(payload.get("histograms"))
        metadata = _parse_metadata(payload.get("metadata"))
        uncertainty = _parse_uncertainty(payload.get("uncertainty"))
        _validate_semantics(histograms, metadata)
        result = GravityAggregateObservation(
            schema_version=GRAVITY_AGGREGATE_SCHEMA_VERSION,
            channel_name=channel_name,
            kind=kind,
            histograms=histograms,
            metadata=metadata,
            uncertainty=uncertainty,
            source_path=source,
            file_sha256=file_sha256,
            content_sha256=declared_checksum,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        _write_audit(
            audit, _rejected_audit(source, file_sha256=file_sha256, error=error)
        )
        raise ValueError(f"invalid aggregate observation {source}: {error}") from error
    _write_audit(audit, result.audit_payload())
    return result


__all__ = [
    "GRAVITY_AGGREGATE_SCHEMA_VERSION",
    "SUPPORTED_AGGREGATE_LIKELIHOODS",
    "SUPPORTED_AGGREGATE_UNITS",
    "GravityAggregateBin",
    "GravityAggregateHistogram",
    "GravityAggregateObservation",
    "GravityAggregateStratum",
    "GravityAggregateUncertainty",
    "load_gravity_aggregate",
]
