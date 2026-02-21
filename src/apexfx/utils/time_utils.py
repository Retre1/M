"""Forex session time handling and timezone utilities."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

import numpy as np


class ForexSession(str, Enum):
    TOKYO = "tokyo"
    LONDON = "london"
    NEW_YORK = "new_york"
    SYDNEY = "sydney"
    OVERLAP_LN = "overlap_ln"
    CLOSED = "closed"


# UTC hour ranges for each session
SESSION_HOURS: dict[ForexSession, tuple[int, int]] = {
    ForexSession.SYDNEY: (21, 6),
    ForexSession.TOKYO: (0, 9),
    ForexSession.LONDON: (7, 16),
    ForexSession.NEW_YORK: (12, 21),
    ForexSession.OVERLAP_LN: (12, 16),
}


def get_active_sessions(utc_hour: int) -> list[ForexSession]:
    """Return all active forex sessions for a given UTC hour."""
    active = []
    for session, (start, end) in SESSION_HOURS.items():
        if session == ForexSession.CLOSED:
            continue
        if start <= end:
            if start <= utc_hour < end:
                active.append(session)
        else:  # Crosses midnight (e.g., Sydney 21-6)
            if utc_hour >= start or utc_hour < end:
                active.append(session)
    return active


def is_forex_market_open(dt: datetime) -> bool:
    """Check if forex market is open (Sunday 21:00 UTC to Friday 21:00 UTC)."""
    utc_dt = dt.astimezone(timezone.utc)
    weekday = utc_dt.weekday()  # 0=Monday
    hour = utc_dt.hour

    if weekday == 4 and hour >= 21:  # Friday after 21:00
        return False
    if weekday == 5:  # Saturday
        return False
    if weekday == 6 and hour < 21:  # Sunday before 21:00
        return False
    return True


def encode_time_features(dt: datetime) -> np.ndarray:
    """Encode time as sinusoidal features: [hour_sin, hour_cos, dow_sin, dow_cos]."""
    utc_dt = dt.astimezone(timezone.utc)
    hour_frac = utc_dt.hour + utc_dt.minute / 60.0
    dow = utc_dt.weekday()

    return np.array([
        np.sin(2 * np.pi * hour_frac / 24.0),
        np.cos(2 * np.pi * hour_frac / 24.0),
        np.sin(2 * np.pi * dow / 7.0),
        np.cos(2 * np.pi * dow / 7.0),
    ], dtype=np.float32)


def get_session_id(dt: datetime) -> int:
    """Return integer session ID for categorical encoding."""
    sessions = get_active_sessions(dt.hour)
    if ForexSession.OVERLAP_LN in sessions:
        return 4
    if ForexSession.NEW_YORK in sessions:
        return 3
    if ForexSession.LONDON in sessions:
        return 2
    if ForexSession.TOKYO in sessions:
        return 1
    if ForexSession.SYDNEY in sessions:
        return 0
    return 5  # off-hours
