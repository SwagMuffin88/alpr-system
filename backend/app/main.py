from __future__ import annotations

import asyncio
import datetime
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()

import app.alpr.main as alpr
from app.sheets import DeduplicatingSheetsWriter, SheetsManager

POLL_INTERVAL_SECONDS = 0.1


def _result_confidence(result: dict[str, Any]) -> float | None:
    confidence = result.get("ocr_confidence")
    if confidence is None:
        confidence = result.get("detection_confidence")
    return float(confidence) if confidence is not None else None


def _format_timestamp(timestamp: datetime.datetime) -> str:
    return (
        f"{timestamp.month}/{timestamp.day}/{timestamp.year} "
        f"{timestamp:%H:%M:%S}"
    )


async def run() -> None:
    request = alpr.StreamRequest(os.getenv("STREAM_URL"))
    writer = DeduplicatingSheetsWriter(SheetsManager())
    await alpr.start_stream(request)
    last_result_seq = 0
    print(f"Started plate detection stream: {request.url}")
    try:
        while True:
            state = await alpr.stream_state()
            if state["status"] == "error":
                raise RuntimeError(state["error"] or "The plate detection stream failed")

            result_seq = state["result_seq"]
            if result_seq != last_result_seq:
                for result in state["results"]:
                    plate = result.get("plate", "")
                    print(f"Plate detected: {plate}")
                    timestamp = _format_timestamp(datetime.datetime.now().astimezone())
                    try:
                        added = await asyncio.to_thread(
                            writer.append_detection,
                            plate,
                            _result_confidence(result),
                            timestamp,
                        )
                    except Exception as exc:
                        print(f"Could not add plate {plate} to the spreadsheet: {exc}")
                    else:
                        if not added:
                            print(f"Skipped recently added plate {plate}")
                last_result_seq = result_seq

            await asyncio.sleep(POLL_INTERVAL_SECONDS)
    finally:
        await alpr.stop_stream()


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("Plate detection stopped")


if __name__ == "__main__":
    main()
