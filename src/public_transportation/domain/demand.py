from __future__ import annotations

from dataclasses import dataclass

from .issues import Issue, Severity, ValidationReport


@dataclass(slots=True, frozen=True)
class ODRecord:
    """
    One OD demand record.

    :param origin_stop_id: Origin stop id.
    :param dest_stop_id: Destination stop id.
    :param time_bin_id: Departure time bin id.
    :param flow: Non-negative flow.
    """
    origin_stop_id: str
    dest_stop_id: str
    time_bin_id: str
    flow: float


@dataclass(slots=True)
class ODDemand:
    """
    Collection of OD demand records.

    :param records: List of OD demand records.
    """
    records: list[ODRecord]

    def validate(self) -> ValidationReport:
        """
        Local validation (referential checks done in Scenario).

        :return: ValidationReport.
        """
        rep = ValidationReport(issues=[])
        for k, r in enumerate(self.records):
            loc = f"demand.records[{k}]"
            if not r.origin_stop_id or not r.dest_stop_id:
                rep.add(Issue(
                    severity=Severity.ERROR,
                    code="DEMAND_OD_EMPTY",
                    message="Origin and destination stop ids must be non-empty.",
                    location=loc,
                ))
            if not r.time_bin_id:
                rep.add(Issue(
                    severity=Severity.ERROR,
                    code="DEMAND_TIMEBIN_EMPTY",
                    message="time_bin_id must be non-empty.",
                    location=loc,
                ))
            if r.flow < 0:
                rep.add(Issue(
                    severity=Severity.ERROR,
                    code="DEMAND_FLOW_NEGATIVE",
                    message="Flow must be non-negative.",
                    location=f"{loc}.flow",
                    context={"flow": r.flow},
                ))
        return rep