from __future__ import annotations

from public_transportation.domain.issues import Issue, Severity, ValidationReport


def test_severity_values_are_stable_strings():
    assert Severity.ERROR.value == "error"
    assert Severity.WARNING.value == "warning"
    assert Severity.INFO.value == "info"


def test_issue_is_immutable_frozen():
    iss = Issue(
        severity=Severity.ERROR,
        code="X",
        message="m",
        location="loc",
        suggestion=None,
        context=None,
    )
    # dataclass(frozen=True) should prevent mutation
    try:
        iss.code = "Y"  # type: ignore[misc]
        assert False, "Issue should be frozen and not allow attribute assignment"
    except Exception as e:
        # TypeError is typical; keep it generic across python versions
        assert type(e).__name__ in {"FrozenInstanceError", "TypeError"}


def test_validation_report_ok_true_when_no_errors():
    rep = ValidationReport(
        issues=[
            Issue(severity=Severity.WARNING, code="W", message="warn", location="x"),
            Issue(severity=Severity.INFO, code="I", message="info", location="y"),
        ]
    )
    assert rep.ok is True


def test_validation_report_ok_false_when_any_error_present():
    rep = ValidationReport(
        issues=[
            Issue(severity=Severity.WARNING, code="W", message="warn", location="x"),
            Issue(severity=Severity.ERROR, code="E", message="err", location="y"),
        ]
    )
    assert rep.ok is False


def test_add_appends_issue():
    rep = ValidationReport(issues=[])
    rep.add(Issue(severity=Severity.INFO, code="I1", message="m1", location="loc1"))
    rep.add(Issue(severity=Severity.WARNING, code="W1", message="m2", location="loc2"))
    assert [i.code for i in rep.issues] == ["I1", "W1"]


def test_extend_appends_issues_in_order():
    rep1 = ValidationReport(
        issues=[
            Issue(severity=Severity.INFO, code="I1", message="m1", location="loc1"),
            Issue(severity=Severity.WARNING, code="W1", message="m2", location="loc2"),
        ]
    )
    rep2 = ValidationReport(
        issues=[
            Issue(severity=Severity.ERROR, code="E1", message="m3", location="loc3"),
        ]
    )

    rep1.extend(rep2)
    assert [i.code for i in rep1.issues] == ["I1", "W1", "E1"]
    assert rep1.ok is False


def test_to_text_renders_location_and_suggestion():
    rep = ValidationReport(
        issues=[
            Issue(
                severity=Severity.ERROR,
                code="STOP_LAT_RANGE",
                message="Latitude out of range.",
                location="stops[S1].lat",
                suggestion="Check CRS.",
                context={"lat": 999},
            )
        ]
    )
    txt = rep.to_text()
    assert "ERROR STOP_LAT_RANGE" in txt
    assert "[stops[S1].lat]" in txt
    assert "Latitude out of range." in txt
    assert "Suggestion: Check CRS." in txt


def test_to_text_omits_location_when_empty():
    rep = ValidationReport(
        issues=[
            Issue(
                severity=Severity.WARNING,
                code="X",
                message="Something.",
                location="",  # explicitly empty
            )
        ]
    )
    txt = rep.to_text()
    assert "WARNING X" in txt
    assert "[" not in txt  # no location brackets when location is empty
    assert "Something." in txt


def test_to_text_respects_max_issues_and_adds_more_line():
    rep = ValidationReport(
        issues=[
            Issue(severity=Severity.INFO, code="I1", message="m1", location="l1"),
            Issue(severity=Severity.INFO, code="I2", message="m2", location="l2"),
            Issue(severity=Severity.INFO, code="I3", message="m3", location="l3"),
        ]
    )
    txt = rep.to_text(max_issues=2)
    lines = txt.splitlines()
    # 2 issues + truncation line
    assert len(lines) == 3
    assert "INFO I1" in lines[0]
    assert "INFO I2" in lines[1]
    assert "... (1 more)" == lines[2]


def test_to_text_max_issues_none_outputs_all():
    rep = ValidationReport(
        issues=[
            Issue(severity=Severity.INFO, code="I1", message="m1", location="l1"),
            Issue(severity=Severity.INFO, code="I2", message="m2", location="l2"),
        ]
    )
    txt = rep.to_text(max_issues=None)
    lines = txt.splitlines()
    assert len(lines) == 2
    assert "INFO I1" in lines[0]
    assert "INFO I2" in lines[1]