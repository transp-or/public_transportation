"""Structural runtime profile for compact OD assignment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from public_transportation.inference.compact_od_assignment_layout import (
    CompactODAssignmentLayout,
)
from public_transportation.inference.od_parameter_layout import ODParameterLayout


@dataclass(frozen=True, slots=True)
class ODAssignmentRuntimeProfile:
    """Counts that explain assignment and estimation runtime structure."""

    num_od_total: int
    num_free_od: int
    num_fixed_od: int
    num_fixed_zero_od: int
    num_fixed_positive_od: int
    assignment_active_od: int
    original_destination_groups: int
    active_destination_groups: int
    removed_destination_groups: int
    od_layout_fingerprint: str | None
    compact_layout_fingerprint: str | None

    def as_dict(self) -> dict[str, int | str | None]:
        """Return a serialization-friendly representation."""
        return asdict(self)

    def format_lines(self) -> tuple[str, ...]:
        """Return concise human-readable reporting lines."""
        return (
            f"OD cells: {self.num_od_total} total, {self.num_free_od} free, "
            f"{self.num_fixed_zero_od} frozen zero, "
            f"{self.num_fixed_positive_od} frozen positive",
            f"Assignment OD vector: {self.assignment_active_od} active cells",
            "Destination groups: "
            f"{self.active_destination_groups}/{self.original_destination_groups} active "
            f"({self.removed_destination_groups} removed)",
        )


def build_od_assignment_runtime_profile(
    *,
    num_od_total: int,
    parameter_layout: ODParameterLayout | None,
    compact_layout: CompactODAssignmentLayout | None,
    artifacts: Any,
    assignment_inputs: Any,
) -> ODAssignmentRuntimeProfile:
    """Build and validate runtime counts from the actual assignment inputs."""
    total = int(num_od_total)
    if parameter_layout is None:
        num_free = total
        num_fixed = num_fixed_zero = num_fixed_positive = 0
        expected_active = total
    else:
        if parameter_layout.num_od_total != total or compact_layout is None:
            raise ValueError("Parameter and compact layouts must match num_od_total.")
        num_free = parameter_layout.num_free
        num_fixed = parameter_layout.num_fixed
        num_fixed_zero = parameter_layout.num_fixed_zero
        num_fixed_positive = parameter_layout.num_fixed_positive
        expected_active = compact_layout.num_active

    actual_active = int(assignment_inputs.od_origin_node.shape[0])
    if actual_active != expected_active:
        raise ValueError(
            "Compact assignment size mismatch: "
            f"expected {expected_active}, got {actual_active}."
        )
    original_groups = int(artifacts.od_groups.group_dest_node.shape[0])
    active_groups = int(assignment_inputs.group_dest_node.shape[0])
    if active_groups > original_groups:
        raise ValueError("Active destination-group count exceeds original count.")

    return ODAssignmentRuntimeProfile(
        num_od_total=total,
        num_free_od=num_free,
        num_fixed_od=num_fixed,
        num_fixed_zero_od=num_fixed_zero,
        num_fixed_positive_od=num_fixed_positive,
        assignment_active_od=actual_active,
        original_destination_groups=original_groups,
        active_destination_groups=active_groups,
        removed_destination_groups=original_groups - active_groups,
        od_layout_fingerprint=(
            None if parameter_layout is None else parameter_layout.fingerprint
        ),
        compact_layout_fingerprint=(
            None if compact_layout is None else compact_layout.fingerprint
        ),
    )
