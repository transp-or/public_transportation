"""Domain objects for the public_transportation package.

This subpackage contains *data structures only* (no plotting, no JAX).

The intent is to keep the domain layer:
- dependency-light,
- easy to validate,
- easy to load from files,
- suitable for future GUI integration.

Visualization lives in ``public_transportation.viz``.
"""

from .demand import ODDemand, ODRecord
from .fixed_demand import FixedODDemand, FixedODRecord, read_fixed_demand_csv
from .issues import Issue, Severity, ValidationReport
from .metadata import Metadata
from .scenario import Scenario
from .stop import Stop
from .stop_time import StopTime
from .time_bin import TimeBin
from .time_of_day import TimeOfDay
from .timetable import Timetable
from .trip import Trip

__all__ = [
    "Issue",
    "Severity",
    "ValidationReport",
    "Metadata",
    "Stop",
    "TimeOfDay",
    "TimeBin",
    "ODRecord",
    "ODDemand",
    "FixedODRecord",
    "FixedODDemand",
    "read_fixed_demand_csv",
    "Trip",
    "StopTime",
    "Timetable",
    "Scenario",
]
