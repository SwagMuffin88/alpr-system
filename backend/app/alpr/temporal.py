from __future__ import annotations

import re
import threading
import time
from collections import Counter, deque
from collections.abc import Iterable
from typing import Any


def normalize_plate(value: str) -> str:
    """Normalize OCR text for temporal matching without changing displayed text."""
    return re.sub(r"[^A-Z0-9]", "", value.upper())


class TemporalPlateFilter:
    """Confirm a plate only after it appears in enough distinct recent frames."""

    def __init__(self, min_hits: int = 4, window_seconds: float = 10.0) -> None:
        self._validate(min_hits, window_seconds)
        self._min_hits = min_hits
        self._window_seconds = window_seconds
        self._events: deque[tuple[float, str]] = deque()
        self._lock = threading.Lock()

    @staticmethod
    def _validate(min_hits: int, window_seconds: float) -> None:
        if not 1 <= min_hits <= 20:
            raise ValueError("Minimum plate hits must be between 1 and 20")
        if not 1 <= window_seconds <= 60:
            raise ValueError("Plate confirmation window must be between 1 and 60 seconds")

    @property
    def min_hits(self) -> int:
        return self._min_hits

    @property
    def window_seconds(self) -> float:
        return self._window_seconds

    def configure(self, min_hits: int) -> None:
        self._validate(min_hits, self._window_seconds)
        with self._lock:
            self._min_hits = min_hits

    def apply(
            self, results: list[dict[str, Any]], now: float | None = None
    ) -> list[dict[str, Any]]:
        timestamp = time.monotonic() if now is None else now
        with self._lock:
            self._prune(timestamp)
            # A duplicate detector box in one frame is still only one temporal hit.
            keys_in_frame = {
                key
                for result in results
                if (key := normalize_plate(str(result.get("plate") or "")))
            }
            self._events.extend((timestamp, key) for key in keys_in_frame)
            counts = Counter(key for _, key in self._events)
            return [
                result
                for result in results
                if (key := normalize_plate(str(result.get("plate") or "")))
                   and counts[key] >= self._min_hits
            ]

    def counts(self, now: float | None = None) -> dict[str, int]:
        timestamp = time.monotonic() if now is None else now
        with self._lock:
            self._prune(timestamp)
            return dict(Counter(key for _, key in self._events))

    def _prune(self, now: float) -> None:
        cutoff = now - self._window_seconds
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()


class TemporalFilterRegistry:
    """Keeps independent short-lived filters for browser webcam sessions."""

    def __init__(self, window_seconds: float = 10.0, session_ttl_seconds: float = 60.0) -> None:
        self.window_seconds = window_seconds
        self.session_ttl_seconds = session_ttl_seconds
        self._sessions: dict[str, tuple[TemporalPlateFilter, float]] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str, min_hits: int) -> TemporalPlateFilter:
        now = time.monotonic()
        with self._lock:
            self._prune(now)
            existing = self._sessions.get(session_id)
            if existing is None:
                plate_filter = TemporalPlateFilter(min_hits, self.window_seconds)
            else:
                plate_filter = existing[0]
                plate_filter.configure(min_hits)
            self._sessions[session_id] = (plate_filter, now)
            return plate_filter

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()

    def _prune(self, now: float) -> None:
        expired = [
            session_id
            for session_id, (_, last_used) in self._sessions.items()
            if now - last_used > self.session_ttl_seconds
        ]
        for session_id in expired:
            del self._sessions[session_id]


def confirmed_keys(results: Iterable[dict[str, Any]]) -> set[str]:
    """Return normalized non-empty plate keys; useful to consumers and tests."""
    return {key for result in results if (key := normalize_plate(str(result.get("plate") or "")))}
