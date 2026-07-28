# tests/conftest.py
from __future__ import annotations

from pathlib import Path

import pytest


def _repo_root(start: Path) -> Path:
    """
    Walk upward to locate the repository root (heuristic).

    We consider the repo root to be the first parent directory containing
    one of: pyproject.toml, .git, src/.

    :param start: Starting path (typically the directory containing this file).
    :return: Path to the repository root.
    :raises RuntimeError: If no suitable root is found.
    """
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists():
            return p
        if (p / ".git").exists():
            return p
        if (p / "src").exists():
            return p
    raise RuntimeError(f"Could not locate repository root from {start}")


def _candidate_case_study_folders(repo_root: Path) -> list[Path]:
    """
    Provide a list of likely locations for the small case study folder.

    Adjust this list as your project evolves; tests will pick the first
    existing folder.

    :param repo_root: Repository root.
    :return: List of candidate folders.
    """
    return [
        # Preferred: tests-managed fixture data
        repo_root / "tests" / "fixtures" / "case_study_small",
        repo_root / "tests" / "fixtures" / "case_study",
        # Docs example folder (often used during development)
        repo_root / "docs" / "source" / "examples" / "network_model" / "data",
        repo_root / "docs" / "source" / "examples" / "network_model" / "case_study_small",
        # Example folder at repo root
        repo_root / "examples" / "network_model" / "data",
        repo_root / "examples" / "network_model" / "case_study_small",
    ]


def _find_case_study_folder(repo_root: Path) -> Path:
    """
    Find an existing case-study folder among candidates.

    :param repo_root: Repository root.
    :return: Path to the first existing case-study folder.
    :raises RuntimeError: If none found.
    """
    for folder in _candidate_case_study_folders(repo_root):
        if folder.exists() and folder.is_dir():
            return folder
    msg = [
        "Could not find a case-study folder in any of the expected locations.",
        "Searched:",
        *[f"  - {p}" for p in _candidate_case_study_folders(repo_root)],
        "",
        "Fix options:",
        "  1) Put your case-study data under tests/fixtures/case_study_small/",
        "  2) Or update tests/conftest.py (_candidate_case_study_folders).",
    ]
    raise RuntimeError("\n".join(msg))


def _has_minimum_scenario_files(folder: Path) -> bool:
    """
    Minimal check that the folder looks like a scenario folder.

    We intentionally keep this permissive (format can be csv/json/parquet).

    :param folder: Scenario folder candidate.
    :return: True if folder contains expected scenario components.
    """
    # Only check "metadata" + "stops" + "time_bins" + "demand".
    # Timetable is optional in principle, but your current examples include it.
    patterns: list[tuple[str, tuple[str, ...]]] = [
        ("metadata", ("metadata.json",)),
        ("stops", ("stops.csv", "stops.json", "stops.parquet")),
        ("time_bins", ("time_bins.csv", "time_bins.json", "time_bins.parquet")),
        ("demand", ("demand.csv", "demand.json", "demand.parquet")),
    ]

    for _, names in patterns:
        if not any((folder / n).exists() for n in names):
            return False
    return True


def _skip_if_no_case_study(folder: Path) -> None:
    """
    Skip tests gracefully if no case study folder is available.

    :param folder: Candidate folder.
    """
    if not (folder.exists() and folder.is_dir() and _has_minimum_scenario_files(folder)):
        pytest.skip(
            f"Case-study folder not available or incomplete: {folder}. "
            "Create tests/fixtures/case_study_small/ or adjust conftest candidates."
        )


# -----------------------
# Repo / paths fixtures
# -----------------------


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """
    Repository root path (heuristically located).

    :return: Path to repository root.
    """
    return _repo_root(Path(__file__).resolve().parent)


@pytest.fixture(scope="session")
def case_study_folder(repo_root: Path) -> Path:
    """
    Path to the small synthetic case-study folder.

    The fixture searches a set of conventional locations.
    Prefer placing a stable fixture under tests/fixtures/case_study_small/.

    :param repo_root: Repository root.
    :return: Path to case study folder.
    """
    folder = _find_case_study_folder(repo_root)
    _skip_if_no_case_study(folder)
    return folder


# -----------------------
# Domain fixtures
# -----------------------


@pytest.fixture(scope="session")
def scenario_small(case_study_folder: Path):
    """
    Load the small scenario from the fixture folder.

    :param case_study_folder: Folder containing metadata/stops/time_bins/demand/(timetable).
    :return: Loaded Scenario instance.
    """
    from public_transportation.domain import Scenario

    return Scenario.from_folder(case_study_folder)


@pytest.fixture(scope="session")
def scenario_small_validation_report(scenario_small):
    """
    Validate the loaded scenario.

    :param scenario_small: Loaded Scenario.
    :return: ValidationReport.
    """
    return scenario_small.validate()


@pytest.fixture(scope="session")
def scenario_small_is_valid(scenario_small_validation_report) -> bool:
    """
    Convenience boolean indicating there are no validation errors.

    :param scenario_small_validation_report: Validation report.
    :return: True iff no ERROR severities are present.
    """
    from public_transportation.domain import Severity

    return all(i.severity != Severity.ERROR for i in scenario_small_validation_report.issues)


# -----------------------
# Assignment fixtures
# -----------------------


@pytest.fixture(scope="session")
def assignment_config_default():
    """
    Provide a default AssignmentConfig for tests.

    This is intentionally minimal. Individual tests can override fields.

    :return: AssignmentConfig instance.
    """
    from public_transportation.assignment.config import AssignmentConfig

    return AssignmentConfig()


@pytest.fixture(scope="session")
def jax_available() -> bool:
    """
    Check whether JAX can be imported.

    :return: True if JAX is importable, else False.
    """
    try:
        import jax  # noqa: F401

        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def require_jax(jax_available: bool):
    """
    Skip the test session (for JAX-dependent tests) if JAX isn't available.

    Usage:
        def test_x(require_jax): ...

    :param jax_available: Whether JAX import succeeded.
    """
    if not jax_available:
        pytest.skip("JAX is not available in this environment.")


@pytest.fixture(scope="function")
def jax_key(require_jax):
    """
    Provide a deterministic JAX PRNG key for tests.

    :param require_jax: Fixture ensuring JAX is available.
    :return: jax.random.PRNGKey(0)
    """
    import jax

    return jax.random.PRNGKey(0)


# -----------------------
# Common small numeric tolerances
# -----------------------


@pytest.fixture(scope="session")
def tol():
    """
    Numeric tolerances for floating-point comparisons.

    :return: Dict with absolute/relative tolerances.
    """
    return {"atol": 1e-6, "rtol": 1e-6}