from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo


@dataclass(slots=True, frozen=True)
class TimeOfDay:
    """
    Time of day with an internal "seconds from midnight" representation.

    This class is designed for:
    - stable machine representation (seconds_from_midnight),
    - easy conversion to/from user-friendly text,
    - optional conversion to timezone-aware datetime given a service date.

    :param seconds_from_midnight: Seconds since 00:00:00. Can exceed 86400 to represent after-midnight service.
    """
    seconds_from_midnight: int

    def __post_init__(self) -> None:
        if self.seconds_from_midnight < 0:
            raise ValueError("seconds_from_midnight must be non-negative.")

    @staticmethod
    def from_hms(h: int, m: int, s: int = 0) -> "TimeOfDay":
        """
        Construct from hours/minutes/seconds. Hours may exceed 23.

        :param h: Hours (>= 0).
        :param m: Minutes (0..59).
        :param s: Seconds (0..59).
        :return: TimeOfDay instance.
        """
        if h < 0:
            raise ValueError("h must be >= 0.")
        if not (0 <= m <= 59):
            raise ValueError("m must be in 0..59.")
        if not (0 <= s <= 59):
            raise ValueError("s must be in 0..59.")
        return TimeOfDay(h * 3600 + m * 60 + s)

    @staticmethod
    def parse(text: str) -> "TimeOfDay":
        """
        Parse a time string like "HH:MM" or "HH:MM:SS". HH may exceed 23.

        :param text: Time string.
        :return: TimeOfDay instance.
        """
        parts = text.strip().split(":")
        if len(parts) not in (2, 3):
            raise ValueError(f"Invalid time format: {text!r}. Expected HH:MM or HH:MM:SS.")
        h = int(parts[0])
        m = int(parts[1])
        s = int(parts[2]) if len(parts) == 3 else 0
        return TimeOfDay.from_hms(h, m, s)

    def to_hms(self) -> tuple[int, int, int]:
        """
        Convert to (hours, minutes, seconds). Hours may exceed 23.

        :return: (h, m, s)
        """
        s = self.seconds_from_midnight
        h = s // 3600
        s = s % 3600
        m = s // 60
        s = s % 60
        return h, m, s

    def to_string(self, *, include_seconds: bool = True) -> str:
        """
        Format as "HH:MM:SS" (default) or "HH:MM".

        :param include_seconds: If False, omit seconds.
        :return: Formatted string.
        """
        h, m, s = self.to_hms()
        if include_seconds:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{h:02d}:{m:02d}"

    def to_datetime(self, service_date: date, tz: str = "Europe/Zurich") -> datetime:
        """
        Convert to timezone-aware datetime, allowing times beyond midnight.

        :param service_date: Service date (local).
        :param tz: IANA timezone name.
        :return: Timezone-aware datetime.
        """
        zone = ZoneInfo(tz)
        base = datetime.combine(service_date, time(0, 0, 0), tzinfo=zone)
        return base + timedelta(seconds=self.seconds_from_midnight)