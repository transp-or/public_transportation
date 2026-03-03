from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .demand import ODDemand, ODRecord
from .issues import Issue, Severity, ValidationReport
from .metadata import Metadata
from .stop import Stop
from .line import Line
from .time_bin import TimeBin
from .time_of_day import TimeOfDay
from .stop_time import StopTime
from .timetable import Timetable
from .feasibility import validate_timetable_feasibility
from .trip import Trip
from .io_utils import (
    coerce_time_columns_to_seconds,
    read_json_dict,
    read_table,
    write_dataclass_json,
    write_table,
)


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
    :param lines: List of public transport lines.
    :param time_bins: List of time bins.
    :param demand: OD demand.
    :param timetable: Optional timetable (trips + stop times).
    """
    metadata: Metadata
    stops: list[Stop]
    lines: list[Line]
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
        for l in self.lines:
            rep.extend(l.validate())
        for b in self.time_bins:
            rep.extend(b.validate())
        rep.extend(self.demand.validate())
        if self.timetable is not None:
            rep.extend(self.timetable.validate(known_stop_ids={s.stop_id for s in self.stops}))

        # Uniqueness checks
        rep.extend(self._validate_unique_ids())

        # Referential integrity checks
        rep.extend(self._validate_references())

        # Feasibility of the time table
        rep.extend(validate_timetable_feasibility(self))

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
        check_unique("lines", [l.line_id for l in self.lines])
        check_unique("time_bins", [b.bin_id for b in self.time_bins])

        return rep

    def _validate_references(self) -> ValidationReport:
        rep = ValidationReport(issues=[])

        stop_ids = {s.stop_id for s in self.stops}
        line_ids = {l.line_id for l in self.lines}
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
            tbid = getattr(r, "time_bin_id", None)
            if tbid is None or str(tbid).strip() == "":
                rep.add(Issue(
                    severity=Severity.ERROR,
                    code="DEMAND_TIMEBIN_MISSING",
                    message="Demand record is missing time_bin_id.",
                    location=loc,
                    suggestion="Set demand.time_bin_id to an existing TimeBin.bin_id.",
                ))
            elif str(tbid) not in time_bin_ids:
                rep.add(Issue(
                    severity=Severity.ERROR,
                    code="DEMAND_TIMEBIN_UNKNOWN",
                    message=f"Unknown time_bin_id: {str(tbid)!r}.",
                    location=loc,
                    suggestion="Fix the demand record to reference an existing time bin, or add the time bin to Scenario.time_bins.",
                    context={"time_bin_id": str(tbid)},
                ))

        # Timetable references (if present)
        if self.timetable is not None:
            for k, tr in enumerate(self.timetable.trips):
                loc = f"timetable.trips[{k}]"
                line_ref = getattr(tr, "line_ref", None)
                if not line_ref or str(line_ref).strip() == "":
                    rep.add(Issue(
                        severity=Severity.ERROR,
                        code="TRIP_LINE_REF_MISSING",
                        message="Trip is missing required line_ref.",
                        location=loc,
                        suggestion="Set trip.line_ref to an existing Line.line_id.",
                        context={"trip_id": getattr(tr, "trip_id", None)},
                    ))
                elif str(line_ref) not in line_ids:
                    rep.add(Issue(
                        severity=Severity.ERROR,
                        code="TRIP_LINE_REF_UNKNOWN",
                        message=f"Unknown line_ref: {line_ref!r}.",
                        location=loc,
                        suggestion="Add the line to Scenario.lines or fix the trip's line_ref.",
                        context={"trip_id": getattr(tr, "trip_id", None), "line_ref": str(line_ref)},
                    ))

        return rep

    # ---------- Convenience lookups ----------
    def get_time_bin(self, bin_id: str) -> TimeBin | None:
        """Return the TimeBin with the given id, or None if not found."""
        bid = str(bin_id)
        for b in self.time_bins:
            if str(b.bin_id) == bid:
                return b
        return None

    def require_time_bin(self, bin_id: str) -> TimeBin:
        """Return the TimeBin with the given id, raising a ValueError if not found."""
        tb = self.get_time_bin(str(bin_id))
        if tb is None:
            raise ValueError(f"Unknown time_bin_id: {bin_id!r}")
        return tb

    def time_bin_interval_min(self, bin_id: str) -> tuple[float, float]:
        """Return (start_min, end_min) for a time bin id."""
        tb = self.require_time_bin(bin_id)
        a = float(tb.start.seconds_from_midnight) / 60.0
        b = float(tb.end.seconds_from_midnight) / 60.0
        return a, b

    def time_bin_interval_s(self, bin_id: str) -> tuple[int, int]:
        """Return (start_s, end_s) for a time bin id."""
        tb = self.require_time_bin(bin_id)
        return int(tb.start.seconds_from_midnight), int(tb.end.seconds_from_midnight)

    # ---------- Folder I/O ----------
    @staticmethod
    def from_folder(folder: str | Path, *, strict: bool = False) -> "Scenario":
        """
        Load a scenario from a folder.

        Expected default filenames:
        - metadata.json
        - stops.(csv|parquet|json)
        - lines.(csv|parquet|json)
        - time_bins.(csv|parquet|json)
        - links.(csv|parquet|json)
        - demand.(csv|parquet|json)
        - trips.(csv|parquet|json) (optional; requires stop_times.* as well)
        - stop_times.(csv|parquet|json) (optional; requires trips.* as well)

        :param folder: Folder path.
        :param strict: If True, run Scenario.validate() after loading and raise a ValueError if any ERROR issues are found.
        :return: Scenario.
        """
        f = Path(folder)

        # ---- Metadata ----
        try:
            md_raw = read_json_dict(f / "metadata.json")
            metadata = Metadata(**md_raw)
        except TypeError as e:
            # Provide a clear error when metadata.json keys do not match Metadata fields.
            # This commonly happens when the file uses different key names (e.g. "name" instead of "title").
            import dataclasses

            expected = [fld.name for fld in dataclasses.fields(Metadata)]
            found = list(md_raw.keys()) if isinstance(md_raw, dict) else []
            raise TypeError(
                "Invalid metadata.json format: keys do not match the Metadata dataclass. "
                f"Expected keys: {expected}. Found keys: {found}. "
                "Fix metadata.json to use the expected keys, or update Metadata accordingly. "
                f"Original error: {e}"
            ) from e

        stops_df = _read_any(f, "stops")
        lines_df = _read_any(f, "lines")
        time_bins_df = _read_any(f, "time_bins")
        demand_df = _read_any(f, "demand")
        trips_df = _read_any(f, "trips", required=False)
        stop_times_df = _read_any(f, "stop_times", required=False)

        # ---- Stops ----
        required_cols = {"stop_id"}
        missing_required = required_cols - set(stops_df.columns)
        if missing_required:
            raise ValueError(
                f"stops table is missing required columns: {sorted(missing_required)}. "
                f"Found columns: {list(stops_df.columns)}. "
                "Expected at least: stop_id. Optional but recommended: name, lat, lon."
            )

        # lat/lon are optional but recommended; provide clear diagnostics if absent
        has_lat = "lat" in stops_df.columns
        has_lon = "lon" in stops_df.columns
        if not (has_lat and has_lon):
            raise ValueError(
                "stops table must contain 'lat' and 'lon' columns. "
                "These are required by Scenario.from_folder to build Stop objects. "
                f"Found columns: {list(stops_df.columns)}. "
                "Fix the input file by adding 'lat' and 'lon' columns "
                "(use dummy values like 0.0 for simple examples if needed)."
            )

        stops = [
            Stop(
                stop_id=str(row["stop_id"]),
                name=str(row.get("name", "")),
                lat=float(row["lat"]),
                lon=float(row["lon"]),
            )
            for _, row in stops_df.iterrows()
        ]

        # ---- Lines ----
        required_cols = {"line_id"}
        missing_required = required_cols - set(lines_df.columns)
        if missing_required:
            raise ValueError(
                f"lines table is missing required columns: {sorted(missing_required)}. "
                f"Found columns: {list(lines_df.columns)}. "
                "Expected at least: line_id. Optional: short_name, long_name, mode, agency_id."
            )

        lines: list[Line] = []
        for k, row in lines_df.iterrows():
            line_id = str(row["line_id"]) if not pd.isna(row["line_id"]) else ""
            if not line_id or line_id.strip() == "":
                raise ValueError(
                    "Invalid value in lines.line_id column. "
                    f"Row index {k} has line_id={row.get('line_id', None)!r}. "
                    "Expected a non-empty string."
                )

            # Map common simple-column conventions to the domain Line model.
            # Preferred columns: short_name, long_name, mode, agency_id.
            # For backward compatibility with very small examples, a column named `name`
            # is interpreted as `short_name`.
            short_name = None
            if "short_name" in lines_df.columns:
                v = row.get("short_name", None)
                short_name = None if pd.isna(v) else str(v)
            elif "name" in lines_df.columns:
                v = row.get("name", None)
                short_name = None if pd.isna(v) else str(v)

            long_name = None
            if "long_name" in lines_df.columns:
                v = row.get("long_name", None)
                long_name = None if pd.isna(v) else str(v)

            mode = None
            if "mode" in lines_df.columns:
                v = row.get("mode", None)
                mode = None if pd.isna(v) else str(v)

            agency_id = None
            if "agency_id" in lines_df.columns:
                v = row.get("agency_id", None)
                agency_id = None if pd.isna(v) else str(v)

            lines.append(
                Line(
                    line_id=line_id,
                    short_name=short_name,
                    long_name=long_name,
                    mode=mode,
                    agency_id=agency_id,
                )
            )

        # ---- Time bins ----
        required_cols = {"bin_id", "start_s", "end_s"}
        missing_required = required_cols - set(time_bins_df.columns)
        if missing_required:
            raise ValueError(
                "time_bins table is missing required columns: "
                f"{sorted(missing_required)}. "
                f"Found columns: {list(time_bins_df.columns)}. "
                "Expected columns: bin_id, start_s, end_s (seconds or 'HH:MM'/'HH:MM:SS'). "
                "Example row:\n"
                "  bin_id,start_s,end_s\n"
                "  TB0,28800,29700\n"
            )

        # Accept either seconds (int) or time strings ("HH:MM" / "HH:MM:SS") in *_s columns.
        time_bins_df = coerce_time_columns_to_seconds(time_bins_df, ["start_s", "end_s"], inplace=False)
        if time_bins_df["start_s"].isna().any() or time_bins_df["end_s"].isna().any():
            bad_rows = time_bins_df[time_bins_df[["start_s", "end_s"]].isna().any(axis=1)].index.tolist()
            raise ValueError(
                "time_bins table contains missing/invalid time values in start_s/end_s after parsing. "
                f"Problematic row indices: {bad_rows}. "
                "Provide seconds-from-midnight (int) or time strings 'HH:MM' / 'HH:MM:SS'."
            )

        time_bins = [
            TimeBin(
                bin_id=str(row["bin_id"]),
                start=TimeOfDay(int(row["start_s"])),
                end=TimeOfDay(int(row["end_s"])),
            )
            for _, row in time_bins_df.iterrows()
        ]


        # ---- Demand ----
        required_cols = {"origin_stop_id", "dest_stop_id", "time_bin_id", "flow"}
        missing_required = required_cols - set(demand_df.columns)
        if missing_required:
            raise ValueError(
                "demand table is missing required columns: "
                f"{sorted(missing_required)}. "
                f"Found columns: {list(demand_df.columns)}. "
                "Expected columns: origin_stop_id, dest_stop_id, time_bin_id, flow. "
                "Example row:\n"
                "  origin_stop_id,dest_stop_id,time_bin_id,flow\n"
                "  O,D,TB0,100\n"
            )

        demand_records: list[ODRecord] = []
        for k, row in demand_df.iterrows():
            try:
                flow_val = float(row["flow"])
            except Exception as e:
                raise ValueError(
                    "Invalid value in demand.flow column. "
                    f"Row index {k} has flow={row.get('flow', None)!r}. "
                    "Expected a numeric value (int/float)."
                ) from e

            demand_records.append(
                ODRecord(
                    origin_stop_id=str(row["origin_stop_id"]),
                    dest_stop_id=str(row["dest_stop_id"]),
                    time_bin_id=str(row["time_bin_id"]),
                    flow=flow_val,
                )
            )

        demand = ODDemand(records=demand_records)

        timetable: Timetable | None = None
        if (trips_df is None) != (stop_times_df is None):
            raise FileNotFoundError(
                "Timetable loading requires both trips.* and stop_times.* to be present (same folder)."
            )
        if trips_df is not None and stop_times_df is not None:
            # ---- Trips ----
            required_cols = {"trip_id", "line_id"}
            missing_required = required_cols - set(trips_df.columns)
            if missing_required:
                raise ValueError(
                    f"trips table is missing required columns: {sorted(missing_required)}. "
                    f"Found columns: {list(trips_df.columns)}. "
                    "Expected at least: trip_id, line_id. Optional: service_id, headsign, direction_id."
                )
            trips = [
                Trip(
                    trip_id=str(row["trip_id"]),
                    line_ref=str(row["line_id"]) if not pd.isna(row["line_id"]) else "",
                    service_id=None if pd.isna(row.get("service_id", None)) else str(row.get("service_id")),
                    headsign=None if pd.isna(row.get("headsign", None)) else str(row.get("headsign")),
                    direction_id=None if pd.isna(row.get("direction_id", None)) else int(row.get("direction_id")),
                )
                for _, row in trips_df.iterrows()
            ]
            for k, tr in enumerate(trips):
                if not tr.line_ref or tr.line_ref.strip() == "":
                    raise ValueError(
                        "trips table contains missing/empty line_id values. "
                        f"Problematic trip at index {k} has trip_id={tr.trip_id!r}. "
                        "Provide a non-empty line_id for every trip."
                    )

            # ---- Stop times ----
            required_cols = {"trip_id", "stop_id", "sequence", "arrival_s", "departure_s"}
            missing_required = required_cols - set(stop_times_df.columns)
            if missing_required:
                raise ValueError(
                    "stop_times table is missing required columns: "
                    f"{sorted(missing_required)}. "
                    f"Found columns: {list(stop_times_df.columns)}. "
                    "Expected columns: trip_id, stop_id, sequence, arrival_s, departure_s (seconds or 'HH:MM'/'HH:MM:SS'). "
                    "Example row:\n"
                    "  trip_id,stop_id,sequence,arrival_s,departure_s\n"
                    "  T1,O,1,28800,28800\n"
                )
            # Accept either seconds (int) or time strings ("HH:MM" / "HH:MM:SS") in *_s columns.
            stop_times_df = coerce_time_columns_to_seconds(stop_times_df, ["arrival_s", "departure_s"], inplace=False)
            if stop_times_df["arrival_s"].isna().any() or stop_times_df["departure_s"].isna().any():
                bad_rows = stop_times_df[stop_times_df[["arrival_s", "departure_s"]].isna().any(axis=1)].index.tolist()
                raise ValueError(
                    "stop_times table contains missing/invalid time values in arrival_s/departure_s after parsing. "
                    f"Problematic row indices: {bad_rows}. "
                    "Provide seconds-from-midnight (int) or time strings 'HH:MM' / 'HH:MM:SS'."
                )
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

        scenario = Scenario(
            metadata=metadata,
            stops=stops,
            lines=lines,
            time_bins=time_bins,
            demand=demand,
            timetable=timetable,
        )

        if strict:
            rep = scenario.validate()
            errors = [it for it in rep.issues if it.severity == Severity.ERROR]
            if errors:
                # Build a readable error message (including suggestions when available)
                lines_out: list[str] = []
                lines_out.append("Scenario validation failed (strict=True).")
                lines_out.append(f"Number of ERROR issues: {len(errors)}")
                for it in errors:
                    msg = f"- [{it.code}] {it.message} ({it.location})"
                    lines_out.append(msg)
                    if getattr(it, "suggestion", None):
                        lines_out.append(f"    suggestion: {it.suggestion}")
                    ctx = getattr(it, "context", None)
                    if isinstance(ctx, dict) and ctx:
                        lines_out.append(f"    context: {ctx}")
                raise ValueError("\n".join(lines_out))

        return scenario

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

        lines_df = pd.DataFrame(
            [
                {
                    "line_id": l.line_id,
                    "short_name": l.short_name,
                    "long_name": l.long_name,
                    "mode": l.mode,
                    "agency_id": l.agency_id,
                }
                for l in self.lines
            ]
        )

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
                    "line_id": getattr(t, "line_ref", None),
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
        write_table(lines_df, f / f"lines.{ext}")
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