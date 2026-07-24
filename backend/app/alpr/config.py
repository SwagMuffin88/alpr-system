from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    detector_model: str = os.getenv("ALPR_DETECTOR_MODEL", "yolo-v9-t-384-license-plate-end2end")
    ocr_model: str = os.getenv("ALPR_OCR_MODEL", "cct-s-v2-global-model")
    detector_confidence: float = float(os.getenv("ALPR_DETECTOR_CONFIDENCE", "0.8"))
    execution_provider: str = os.getenv("ALPR_EXECUTION_PROVIDER", "CPUExecutionProvider")
    jpeg_quality: int = min(100, max(40, int(os.getenv("ALPR_JPEG_QUALITY", "88"))))


settings = Settings()
