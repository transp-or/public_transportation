"""Errors raised while configuring structural-zero preprocessing."""


class StructuralZeroConfigError(ValueError):
    """A TOML configuration is missing, unknown, or semantically invalid."""


class StructuralZeroConflictError(ValueError):
    """Existing nonzero fixed demand conflicts with detected structural zeros."""

    def __init__(self, conflicts: tuple[tuple[str, str, str, float], ...]) -> None:
        self.conflicts = conflicts
        examples = ", ".join(repr(item) for item in conflicts[:5])
        suffix = "" if len(conflicts) <= 5 else f" (+{len(conflicts) - 5} more)"
        super().__init__(
            "Existing fixed demand assigns a nonzero value to "
            f"{len(conflicts)} structural-zero cell(s): {examples}{suffix}."
        )
