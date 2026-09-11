"""Command-line entry point for ``python -m public_transportation...``."""

from .cli import main


if __name__ == "__main__":  # pragma: no cover - exercised through CLI tests.
    raise SystemExit(main())
