from __future__ import annotations

from dataclasses import dataclass

from .issues import Issue, Severity, ValidationReport


@dataclass(slots=True)
class Stop:
    """
    A geocoded stop.

    :param stop_id: Unique stop identifier.
    :param name: Human-readable stop name.
    :param lat: Latitude in degrees [-90, 90].
    :param lon: Longitude in degrees [-180, 180].
    """
    stop_id: str
    name: str
    lat: float
    lon: float

    def validate(self) -> ValidationReport:
        """
        Validate stop fields.

        :return: ValidationReport with issues.
        """
        rep = ValidationReport(issues=[])

        if not self.stop_id:
            rep.add(Issue(
                severity=Severity.ERROR,
                code="STOP_ID_EMPTY",
                message="Stop id is empty.",
                location="stops[].stop_id",
                suggestion="Provide a non-empty stop_id.",
            ))

        if not self.name:
            rep.add(Issue(
                severity=Severity.WARNING,
                code="STOP_NAME_EMPTY",
                message="Stop name is empty.",
                location=f"stops[{self.stop_id}].name",
                suggestion="Provide a readable stop name.",
            ))

        if not (-90.0 <= float(self.lat) <= 90.0):
            rep.add(Issue(
                severity=Severity.ERROR,
                code="STOP_LAT_RANGE",
                message=f"Latitude {self.lat} is out of range [-90, 90].",
                location=f"stops[{self.stop_id}].lat",
                suggestion="Check coordinate reference system and units.",
                context={"lat": self.lat},
            ))

        if not (-180.0 <= float(self.lon) <= 180.0):
            rep.add(Issue(
                severity=Severity.ERROR,
                code="STOP_LON_RANGE",
                message=f"Longitude {self.lon} is out of range [-180, 180].",
                location=f"stops[{self.stop_id}].lon",
                suggestion="Check coordinate reference system and units.",
                context={"lon": self.lon},
            ))

        return rep