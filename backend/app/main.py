import datetime
import os
from dotenv import load_dotenv
from app.sheets import SheetsManager

load_dotenv()

SPREADSHEET_ID =os.getenv("SPREADSHEET_ID")
CREDENTIALS_FILE = os.getenv("CREDENTIALS_FILE")


def main():
    sheets = SheetsManager()

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sheets.append_detection(plate="777ENV", confidence=0.99, timestamp=now)

if __name__ == "__main__":
    main()