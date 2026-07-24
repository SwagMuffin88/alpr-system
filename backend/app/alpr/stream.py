from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any
from urllib.parse import urlparse

import cv2

from app.alpr.engine import AlprEngine, ProcessedFrame
from app.alpr.temporal import TemporalPlateFilter


class StreamManager:
    """One capture thread plus one latest-frame inference thread for a single URL stream."""

    def __init__(self, engine: AlprEngine) -> None:
        self.engine = engine
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._capture: cv2.VideoCapture | None = None
        self._capture_thread: threading.Thread | None = None
        self._process_thread: threading.Thread | None = None
        self._frame: Any = None
        self._captured_seq = 0
        self._processed_seq = 0
        self._processed_count = 0
        self._dropped_count = 0
        self._processed: ProcessedFrame | None = None
        self._status = "idle"
        self._error: str | None = None
        self._url: str | None = None
        self._max_fps = 5.0
        self._plate_filter = TemporalPlateFilter()
        self._started_at: float | None = None
        self._inference_samples: deque[float] = deque(maxlen=120)
        self._pipeline_samples: deque[float] = deque(maxlen=120)

    def start(self, url: str, max_fps: float = 5.0, min_plate_hits: int = 4) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Stream URL must be an absolute http:// or https:// URL")
        if not 0 <= max_fps <= 60:
            raise ValueError("Maximum inference FPS must be between 0 and 60")
        if not 1 <= min_plate_hits <= 20:
            raise ValueError("Minimum plate hits must be between 1 and 20")
        self.stop()
        with self._lock:
            self._stop = threading.Event()
            self._status = "connecting"
            self._error = None
            self._url = url
            self._max_fps = max_fps
            self._plate_filter = TemporalPlateFilter(min_hits=min_plate_hits)
            self._started_at = time.monotonic()
            self._frame = None
            self._captured_seq = 0
            self._processed_seq = 0
            self._processed_count = 0
            self._dropped_count = 0
            self._processed = None
            self._inference_samples.clear()
            self._pipeline_samples.clear()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._capture_thread.start()
        self._process_thread.start()

    def stop(self) -> None:
        self._stop.set()
        capture = self._capture
        if capture is not None:
            capture.release()
        for thread in (self._capture_thread, self._process_thread):
            if thread and thread.is_alive():
                thread.join(timeout=1.5)
        with self._lock:
            self._capture = None
            if self._status != "idle":
                self._status = "idle"

    def _capture_loop(self) -> None:
        capture = cv2.VideoCapture(self._url or "")
        self._capture = capture
        if not capture.isOpened():
            self._fail("Could not open stream. Check the URL and OpenCV/FFmpeg codec support.")
            return
        with self._lock:
            self._status = "running"
        while not self._stop.is_set():
            ok, frame = capture.read()
            if not ok:
                self._fail("The stream stopped returning frames.")
                break
            with self._lock:
                self._frame = frame
                self._captured_seq += 1
        capture.release()

    def _process_loop(self) -> None:
        last_seq = 0
        next_allowed_at = 0.0
        while not self._stop.wait(0.001):
            now = time.monotonic()
            if self._max_fps > 0 and now < next_allowed_at:
                self._stop.wait(min(next_allowed_at - now, 0.05))
                continue
            with self._lock:
                seq = self._captured_seq
                frame = self._frame
            if frame is None or seq == last_seq:
                continue
            skipped = max(0, seq - last_seq - 1)
            last_seq = seq
            inference_started_at = time.monotonic()
            try:
                processed = self.engine.process(frame, self._plate_filter.apply)
            except Exception as exc:  # model and codec errors need to reach the UI
                self._fail(f"Inference failed: {exc}")
                return
            with self._lock:
                self._processed = processed
                self._processed_seq = seq
                self._processed_count += 1
                self._dropped_count += skipped
                self._inference_samples.append(processed.inference_ms)
                self._pipeline_samples.append(processed.total_ms)
            if self._max_fps > 0:
                next_allowed_at = inference_started_at + (1 / self._max_fps)

    def _fail(self, message: str) -> None:
        with self._lock:
            self._status = "error"
            self._error = message
        self._stop.set()

    def state(self) -> dict[str, Any]:
        with self._lock:
            uptime = time.monotonic() - self._started_at if self._started_at else 0
            samples = list(self._inference_samples)
            pipeline_samples = list(self._pipeline_samples)
            processed = self._processed
            captured = self._captured_seq
            processed_seq = self._processed_seq
            return {
                "status": self._status,
                "error": self._error,
                "url": self._url,
                "max_inference_fps": self._max_fps,
                "plate_filter": {
                    "min_hits": self._plate_filter.min_hits,
                    "window_seconds": self._plate_filter.window_seconds,
                    "candidate_counts": self._plate_filter.counts(),
                },
                "captured_frames": captured,
                "processed_frames": self._processed_count,
                "dropped_frames": self._dropped_count,
                "capture_fps": round(captured / uptime, 2) if uptime else 0,
                "processing_fps": (
                    round(1000 / (sum(pipeline_samples) / len(pipeline_samples)), 2)
                    if pipeline_samples
                    else 0
                ),
                "last_inference_ms": round(samples[-1], 2) if samples else None,
                "result_seq": processed_seq,
                "results": processed.results if processed else [],
                "raw_results": (
                    processed.raw_results
                    if processed and processed.raw_results is not None
                    else (processed.results if processed else [])
                ),
                "frame_metrics": processed.metrics() if processed else None,
            }

    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._processed.jpeg if self._processed else None
