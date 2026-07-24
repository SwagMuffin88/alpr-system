from __future__ import annotations

import os.path
import threading
import time
from collections.abc import Callable

from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from app.alpr.temporal import normalize_plate

load_dotenv()

SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

class SheetsManager:
    def __init__(self, spreadsheet_id: str = None, credentials_path: str = None):
        self.spreadsheet_id = spreadsheet_id or os.getenv("SPREADSHEET_ID")
        self.credentials_path = credentials_path or os.getenv("CREDENTIALS_FILE", "credentials.json")

        if not self.spreadsheet_id:
            raise ValueError("SPREADSHEET_ID not found!")

        self.credentials = None
        self.authenticate()
        self.service = build("sheets", "v4", credentials=self.credentials)

    def authenticate(self):
        if os.path.exists('token.json'):
            self.credentials = Credentials.from_authorized_user_file('token.json', SCOPES)
        if not self.credentials or not self.credentials.valid:
            if self.credentials and self.credentials.expired and self.credentials.refresh_token:
                self.credentials.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(self.credentials_path, SCOPES)
                self.credentials = flow.run_local_server(port=0)
            with open("token.json", "w") as token:
                token.write(self.credentials.to_json())

    def append_detection(self, plate: str, confidence: float | None, timestamp: str) -> dict:
        """Add a detected license plate number to the spreadsheet"""
        sheet = self.service.spreadsheets()
        formatted_confidence = "" if confidence is None else f"{confidence:.2f}"
        body = {"values": [[timestamp, plate, formatted_confidence]]}

        result = (
            sheet.values()
            .append(
                spreadsheetId=self.spreadsheet_id,
                range="Test_sheet!A1",
                valueInputOption="USER_ENTERED",
                body=body,
            )
            .execute()
        )
        print(f"Added plate {plate} to the spreadsheet")
        return result


class DeduplicatingSheetsWriter:
    """Append plates once per rolling local time window."""

    def __init__(
            self,
            sheets: SheetsManager,
            window_seconds: float = 60.0,
            clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if window_seconds <= 0:
            raise ValueError("Duplicate window must be greater than zero seconds")
        self.sheets = sheets
        self.window_seconds = window_seconds
        self._clock = clock
        self._last_added: dict[str, float] = {}
        self._lock = threading.Lock()

    def append_detection(
            self,
            plate: str,
            confidence: float | None,
            timestamp: str,
    ) -> bool:
        """Append a detection unless the normalized plate was added recently."""
        key = normalize_plate(plate)
        if not key:
            return False

        now = self._clock()
        with self._lock:
            self._prune(now)
            if key in self._last_added:
                return False

            # Only remember successful writes, so a transient Sheets failure can
            # be retried when the plate is seen in a later frame.
            self.sheets.append_detection(plate, confidence, timestamp)
            self._last_added[key] = now
            return True

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        expired = [
            plate
            for plate, added_at in self._last_added.items()
            if added_at <= cutoff
        ]
        for plate in expired:
            del self._last_added[plate]
