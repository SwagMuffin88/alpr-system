from __future__ import annotations

import statistics
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from app.alpr.config import Settings, settings


def _value(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
        if isinstance(obj, dict) and name in obj:
            return obj[name]
    return default


def _confidence(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        return float(statistics.fmean(float(item) for item in value)) if len(value) else None
    return float(value)


def serialize_results(results: list[Any]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for result in results:
        detection = _value(result, "detection")
        bbox = _value(detection, "bounding_box", "bbox")
        ocr = _value(result, "ocr")
        serialized.append(
            {
                "bbox": {
                    "x1": int(_value(bbox, "x1", default=0)),
                    "y1": int(_value(bbox, "y1", default=0)),
                    "x2": int(_value(bbox, "x2", default=0)),
                    "y2": int(_value(bbox, "y2", default=0)),
                },
                "label": str(_value(detection, "class_id", "label", "class_name", default="plate")),
                "detection_confidence": _confidence(
                    _value(detection, "confidence", "score", default=None)
                ),
                "plate": str(_value(ocr, "text", default="")) if ocr is not None else "",
                "ocr_confidence": _confidence(_value(ocr, "confidence", default=None)),
                "country": _value(ocr, "region", default=None),
                "country_confidence": _confidence(_value(ocr, "region_confidence", default=None)),
            }
        )
    return serialized


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def annotate(frame: np.ndarray, results: list[dict[str, Any]]) -> np.ndarray:
    image = frame.copy()
    height, width = image.shape[:2]
    scale = min(0.85, max(0.46, width / 1400))
    thickness = 2 if width >= 720 else 1

    for result in results:
        box = result["bbox"]
        x1 = min(width - 1, max(0, box["x1"]))
        y1 = min(height - 1, max(0, box["y1"]))
        x2 = min(width - 1, max(x1 + 1, box["x2"]))
        y2 = min(height - 1, max(y1 + 1, box["y2"]))
        cv2.rectangle(image, (x1, y1), (x2, y2), (65, 230, 125), thickness)

        plate = result["plate"] or "unreadable"
        lines = [
            f"{plate}  OCR {_percent(result['ocr_confidence'])}",
            f"det {_percent(result['detection_confidence'])}",
        ]
        if result["country"]:
            lines.append(f"{result['country']}  country {_percent(result['country_confidence'])}")

        line_sizes = [
            cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0] for line in lines
        ]
        line_height = max(size[1] for size in line_sizes) + 8
        panel_width = max(size[0] for size in line_sizes) + 14
        panel_height = line_height * len(lines) + 6
        top = y1 - panel_height
        if top < 0:
            top = min(height - panel_height, y2 + 3)
        left = min(max(0, x1), max(0, width - panel_width))
        cv2.rectangle(
            image,
            (left, top),
            (min(width - 1, left + panel_width), min(height - 1, top + panel_height)),
            (12, 18, 24),
            -1,
        )
        for index, line in enumerate(lines):
            baseline = top + 5 + (index + 1) * line_height - 5
            cv2.putText(
                image,
                line,
                (left + 7, baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                (245, 250, 247),
                thickness,
                cv2.LINE_AA,
            )
    return image


def encode_jpeg(frame: np.ndarray, quality: int = 88) -> bytes:
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("Could not encode the annotated image")
    return encoded.tobytes()


def decode_image(data: bytes) -> np.ndarray:
    if not data:
        raise ValueError("The uploaded image is empty")
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("The uploaded file is not a supported image")
    return frame


@dataclass
class ProcessedFrame:
    jpeg: bytes
    results: list[dict[str, Any]]
    inference_ms: float
    total_ms: float
    width: int
    height: int
    raw_results: list[dict[str, Any]] | None = None

    def metrics(self) -> dict[str, Any]:
        raw_results = self.raw_results if self.raw_results is not None else self.results
        return {
            "inference_ms": round(self.inference_ms, 2),
            "total_ms": round(self.total_ms, 2),
            "inference_fps": round(1000 / self.inference_ms, 2) if self.inference_ms else None,
            "pipeline_fps": round(1000 / self.total_ms, 2) if self.total_ms else None,
            "width": self.width,
            "height": self.height,
            "suppressed_detections": max(0, len(raw_results) - len(self.results)),
        }

    def response(self) -> dict[str, Any]:
        return {
            # disabled because no image preview yet.
            # "image": "data:image/jpeg;base64," + base64.b64encode(self.jpeg).decode("ascii"),
            "results": self.results,
            "raw_results": self.raw_results if self.raw_results is not None else self.results,
            "metrics": self.metrics(),
        }


class AlprEngine:
    """Lazily loads FastALPR and serializes access to the model sessions."""

    def __init__(self, config: Settings = settings) -> None:
        self.config = config
        self._model: Any = None
        self._lock = threading.Lock()
        self._load_ms: float | None = None

    @property
    def ready(self) -> bool:
        return self._model is not None

    @property
    def load_ms(self) -> float | None:
        return self._load_ms

    def _get_model(self) -> Any:
        if self._model is None:
            started = time.perf_counter()
            from fast_alpr import ALPR

            providers = (
                None
                if self.config.execution_provider.lower() == "auto"
                else [self.config.execution_provider]
            )
            self._model = ALPR(
                detector_model=self.config.detector_model,
                ocr_model=self.config.ocr_model,
                detector_conf_thresh=self.config.detector_confidence,
                detector_providers=providers,
                ocr_providers=providers,
            )
            self._load_ms = (time.perf_counter() - started) * 1000
        return self._model

    def infer(self, frame: np.ndarray) -> tuple[list[dict[str, Any]], float]:
        with self._lock:
            started = time.perf_counter()
            raw = self._get_model().predict(frame)
            elapsed_ms = (time.perf_counter() - started) * 1000
        return serialize_results(raw), elapsed_ms

    def process(
            self,
            frame: np.ndarray,
            result_filter: Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None = None,
    ) -> ProcessedFrame:
        started = time.perf_counter()
        raw_results, inference_ms = self.infer(frame)
        results = result_filter(raw_results) if result_filter else raw_results
        # disabled because we don't have preview yet.
        rendered = None  # annotate(frame, results)
        jpeg = None  # encode_jpeg(rendered, self.config.jpeg_quality)
        total_ms = (time.perf_counter() - started) * 1000
        return ProcessedFrame(
            jpeg=jpeg,
            results=results,
            inference_ms=inference_ms,
            total_ms=total_ms,
            width=frame.shape[1],
            height=frame.shape[0],
            raw_results=raw_results,
        )
