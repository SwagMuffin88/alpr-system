from __future__ import annotations

import asyncio
import json
import os

from dotenv import load_dotenv

load_dotenv()

import app.alpr.main as alpr

POLL_INTERVAL_SECONDS = 0.1


async def run() -> None:
    request = alpr.StreamRequest(os.getenv("STREAM_URL"))
    await alpr.start_stream(request)
    last_result_seq = 0
    print(f"Started plate detection stream: {request.url}", )
    try:
        while True:
            state = await alpr.stream_state()
            if state["status"] == "error":
                raise RuntimeError(state["error"] or "The plate detection stream failed")

            result_seq = state["result_seq"]
            if result_seq != last_result_seq:
                for result in state["results"]:
                    print(f"Plate detected: {json.dumps(result, ensure_ascii=False, default=str)}")
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
