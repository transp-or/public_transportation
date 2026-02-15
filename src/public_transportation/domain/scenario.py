from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .demand import ODDemand, ODRecord
from .issues import Issue, Severity, ValidationReport
from .metadata import Metadata
from .stop import Stop
from .time_bin import TimeBin
from .time_of_day import TimeOfDay
from .stop_time import StopTime
from .timetable import Timetable
from .trip import Trip
from .io_utils import read_table, read_json_dict, write_dataclass_json, write_table


@dataclass(slots=True)
class Scenario:
    """
    Domain-level scenario container (no JAX).

    Provides:
    - storage of modeling elements,
    - local + cross-element validation,
    - flexible load/save from a folder.

    :param metadata: Scenario metadata.
    :param stops: List of stops.
    :param time_bins: List of time bins.
    :param demand: OD demand.
    :param timetable: Optional timetable (trips + stop times).
    """
    metadata: Metadata
    stops: list[Stop]
    time_bins: list[TimeBin]
    demand: ODDemand
    timetable: Timetable | None = None

    # ---------- Validation ----------
    def validate(self) -> ValidationReport:
        """
        Validate internal consistency (IDs, references, basic domain rules).

        :return: ValidationReport.
        """
        rep = ValidationReport(issues=[])

        # Local validations
        for s in self.stops:
            rep.extend(s.validate())
        for b in self.time_bins:
            rep.extend(b.validate())
        rep.extend(self.demand.validate())
        if self.timetable is not None:
            rep.extend(self.timetable.validate(known_stop_ids={s.stop_id for s in self.stops}))

        # Uniqueness checks
        rep.extend(self._validate_unique_ids())

        # Referential integrity checks
        rep.extend(self._validate_references())

        return rep

    def _validate_unique_ids(self) -> ValidationReport:
        rep = ValidationReport(issues=[])

        def check_unique(name: str, ids: list[str]) -> None:
            seen: set[str] = set()
            for i in ids:
                if i in seen:
                    rep.add(Issue(
                        severity=Severity.ERROR,
                        code="DUPLICATE_ID",
                        message=f"Duplicate {name} id: {i!r}.",
                        location=name,
                        suggestion="Ensure identifiers are unique.",
                        context={"id": i},
                    ))
                else:
                    seen.add(i)

        check_unique("stops", [s.stop_id for s in self.stops])
        check_unique("time_bins", [b.bin_id for b in self.time_bins])

        return rep

    def _validate_references(self) -> ValidationReport:
        rep = ValidationReport(issues=[])

        stop_ids = {s.stop_id for s in self.stops}
        time_bin_ids = {b.bin_id for b in self.time_bins}

        # Demand references
        for k, r in enumerate(self.demand.records):
            loc = f"demand.records[{k}]"
            if r.origin_stop_id not in stop_ids:
                rep.add(Issue(
                    severity=Severity.ERROR,
                    code="DEMAND_ORIGIN_UNKNOWN",
                    message=f"Unknown origin_stop_id: {r.origin_stop_id!r}.",
                    location=loc,
                ))
            if r.dest_stop_id not in stop_ids:
                rep.add(Issue(
                    severity=Severity.ERROR,
                    code="DEMAND_DEST_UNKNOWN",
                    message=f"Unknown dest_stop_id: {r.dest_stop_id!r}.",
                    location=loc,
                ))
            if r.time_bin_id not in time_bin_ids:
                rep.add(Issue(
                    severity=Severity.ERROR,
                    code="DEMAND_TIMEBIN_UNKNOWN",
                    message=f"Unknown time_bin_id: {r.time_bin_id!r}.",
                    location=loc,
                ))



        return rep

    # ---------- Folder I/O ----------
    @staticmethod
    def from_folder(folder: str | Path) -> "Scenario":
        """
        Load a scenario from a folder.

        Expected default filenames:
        - metadata.json
        - stops.(csv|parquet|json)
        - time_bins.(csv|parquet|json)
        - links.(csv|parquet|json)
        - demand.(csv|parquet|json)
        - trips.(csv|parquet|json) (optional; requires stop_times.* as well)
        - stop_times.(csv|parquet|json) (optional; requires trips.* as well)

        :param folder: Folder path.
        :return: Scenario.
        """
        f = Path(folder)

        metadata = Metadata(**read_json_dict(f / "metadata.json"))

        stops_df = _read_any(f, "stops")
        time_bins_df = _read_any(f, "time_bins")
        demand_df = _read_any(f, "demand")
        trips_df = _read_any(f, "trips", required=False)
        stop_times_df = _read_any(f, "stop_times", required=False)

        stops = [
            Stop(
                stop_id=str(row["stop_id"]),
                name=str(row.get("name", "")),
                lat=float(row["lat"]),
                lon=float(row["lon"]),
            )
            for _, row in stops_df.iterrows()
        ]

        time_bins = [
            TimeBin(
                bin_id=str(row["bin_id"]),
                start=TimeOfDay(int(row["start_s"])),
                end=TimeOfDay(int(row["end_s"])),
            )
            for _, row in time_bins_df.iterrows()
        ]


        demand = ODDemand(records=[
            ODRecord(
                origin_stop_id=str(row["origin_stop_id"]),
                dest_stop_id=str(row["dest_stop_id"]),
                time_bin_id=str(row["time_bin_id"]),
                flow=float(row["flow"]),
            )
            for _, row in demand_df.iterrows()
        ])

        timetable: Timetable | None = None
        if (trips_df is None) != (stop_times_df is None):
            raise FileNotFoundError(
                "Timetable loading requires both trips.* and stop_times.* to be present (same folder)."
            )
        if trips_df is not None and stop_times_df is not None:
            trips = [
                Trip(
                    trip_id=str(row["trip_id"]),
                    line_id=None if pd.isna(row.get("line_id", None)) else str(row.get("line_id")),
                    service_id=None if pd.isna(row.get("service_id", None)) else str(row.get("service_id")),
                    headsign=None if pd.isna(row.get("headsign", None)) else str(row.get("headsign")),
                    direction_id=None if pd.isna(row.get("direction_id", None)) else int(row.get("direction_id")),
                )
                for _, row in trips_df.iterrows()
            ]

            stop_times = [
                StopTime(
                    trip_id=str(row["trip_id"]),
                    stop_id=str(row["stop_id"]),
                    sequence=int(row["sequence"]),
                    arrival=TimeOfDay(int(row["arrival_s"])),
                    departure=TimeOfDay(int(row["departure_s"])),
                )
                for _, row in stop_times_df.iterrows()
            ]

            timetable = Timetable(trips=trips, stop_times=stop_times)

        return Scenario(
            metadata=metadata,
            stops=stops,
            time_bins=time_bins,
            demand=demand,
            timetable=timetable,
        )

    def to_folder(self, folder: str | Path, *, table_format: str = "csv") -> None:
        """
        Save scenario to a folder.

        :param folder: Output folder.
        :param table_format: "csv", "parquet", or "json" for tabular files.
        """
        f = Path(folder)
        f.mkdir(parents=True, exist_ok=True)

        write_dataclass_json(self.metadata, f / "metadata.json")

        stops_df = pd.DataFrame([{
            "stop_id": s.stop_id,
            "name": s.name,
            "lat": s.lat,
            "lon": s.lon,
        } for s in self.stops])

        time_bins_df = pd.DataFrame([{
            "bin_id": b.bin_id,
            "start_s": b.start.seconds_from_midnight,
            "end_s": b.end.seconds_from_midnight,
            "start_str": b.start.to_string(include_seconds=False),
            "end_str": b.end.to_string(include_seconds=False),
        } for b in self.time_bins])


        demand_df = pd.DataFrame([{
            "origin_stop_id": r.origin_stop_id,
            "dest_stop_id": r.dest_stop_id,
            "time_bin_id": r.time_bin_id,
            "flow": r.flow,
        } for r in self.demand.records])

        trips_df = None
        stop_times_df = None
        if self.timetable is not None:
            trips_df = pd.DataFrame([
                {
                    "trip_id": t.trip_id,
                    "line_id": t.line_id,
                    "service_id": t.service_id,
                    "headsign": t.headsign,
                    "direction_id": t.direction_id,
                }
                for t in self.timetable.trips
            ])

            stop_times_df = pd.DataFrame([
                {
                    "trip_id": st.trip_id,
                    "stop_id": st.stop_id,
                    "sequence": st.sequence,
                    "arrival_s": st.arrival.seconds_from_midnight,
                    "departure_s": st.departure.seconds_from_midnight,
                    "arrival_str": st.arrival.to_string(include_seconds=True),
                    "departure_str": st.departure.to_string(include_seconds=True),
                }
                for st in self.timetable.stop_times
            ])

        ext = _ext_from_format(table_format)
        write_table(stops_df, f / f"stops.{ext}")
        write_table(time_bins_df, f / f"time_bins.{ext}")
        write_table(demand_df, f / f"demand.{ext}")
        if trips_df is not None and stop_times_df is not None:
            write_table(trips_df, f / f"trips.{ext}")
            write_table(stop_times_df, f / f"stop_times.{ext}")


def _ext_from_format(fmt: str) -> str:
    fmt = fmt.strip().lower()
    if fmt in {"csv", "parquet", "json"}:
        return fmt
    raise ValueError("table_format must be one of: csv, parquet, json.")


def _read_any(folder: Path, stem: str, *, required: bool = True) -> pd.DataFrame | None:
    for ext in ("csv", "parquet", "json"):
        p = folder / f"{stem}.{ext}"
        if p.exists():
            return read_table(p)
    if required:
        raise FileNotFoundError(f"Could not find {stem}.(csv|parquet|json) in {folder}")
    return None