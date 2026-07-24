from __future__ import annotations

import asyncio

from app.alpr.engine import AlprEngine
from app.alpr.stream import StreamManager
from app.alpr.temporal import TemporalFilterRegistry

engine = AlprEngine()
streams = StreamManager(engine)
webcam_filters = TemporalFilterRegistry()


class StreamRequest:
    url: str
    max_fps: float
    min_plate_hits: int

    def __init__(self, url: str, max_fps=5, min_plate_hits=4):
        self.url = url
        self.max_fps = max_fps
        self.min_plate_hits = min_plate_hits


async def start_stream(request: StreamRequest) -> dict:
    try:
        await asyncio.to_thread(
            streams.start,
            request.url,
            request.max_fps,
            request.min_plate_hits,
        )
        return streams.state()
    except ValueError as exc:
        raise RuntimeError(exc)


async def stop_stream() -> dict:
    await asyncio.to_thread(streams.stop)
    return streams.state()


async def stream_state() -> dict:
    return streams.state()
