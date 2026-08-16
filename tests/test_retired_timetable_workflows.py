"""Safeguards for the retired timetable-journey workflow namespace."""

from __future__ import annotations

import importlib
import sys

import pytest

import public_transportation.case_study as case_study
import public_transportation.inference as inference
import public_transportation.preprocessing as preprocessing


def test_public_namespaces_do_not_expose_retired_routing_symbols():
    retired_names = {
        "run_" + "rap" + "tor_query",
        "run_" + "rap" + "tor_range_query",
        "R" + "aptorResult",
    }
    for namespace in (inference, preprocessing):
        assert retired_names.isdisjoint(vars(namespace))


def test_retired_backend_modules_and_configuration_are_not_importable():
    retired_package = "reduced" + "_od"
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(
            "public_transportation.preprocessing."
            + retired_package
            + "."
            + "rap"
            + "tor"
        )
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("public_transportation.case_study.config")


def test_retired_case_study_workflow_fails_explicitly():
    with pytest.raises(
        case_study.RetiredCaseStudyWorkflowError,
        match="generic timetable-journey case-study workflow is retired",
    ):
        case_study.require_case_study_workflow()


def test_direct_scheduled_imports_do_not_load_retired_modules():
    before = set(sys.modules)
    importlib.import_module(
        "public_transportation.inference.direct_scheduled_temporal_builder"
    )
    importlib.import_module(
        "public_transportation.inference.fixed_routing_measurement_operator"
    )
    loaded = set(sys.modules) - before
    retired_package = "." + "reduced" + "_od"
    assert not any(
        retired_package + "." in module_name
        or module_name.endswith(retired_package)
        for module_name in loaded
    )
