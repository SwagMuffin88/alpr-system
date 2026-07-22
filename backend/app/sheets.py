import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from dotenv import load_dotenv

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

    def append_detection(self, plate: str, confidence: float, timestamp: str):
        """Add a detected license plate number to the spreadsheet"""
        try:
            sheet = self.service.spreadsheets()
            values = [[timestamp, plate, f"{confidence:.2f}"]]
            body = {'values': values}

            result = sheet.values().append(
                spreadsheetId=self.spreadsheet_id,
                range="Test_sheet!A1",
                valueInputOption="USER_ENTERED",
                body=body
            ).execute()
            print("Data successfully added to the spreadsheet")
            return result
        except HttpError as err:
            print(f"There was a problem with Google Sheets API: {err}")