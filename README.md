# alpr-system
Automatic license plate recognition solution for ARK event

[Google Sheets API Docs](https://developers.google.com/workspace/sheets/api/guides/concepts)

[FastALPR](https://github.com/ankandrew/fast-alpr)

### Sõltuvuste paigaldamine
```
cd backend
python -m venv venv
# Windows: venv\Scripts\activate | Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
```

### Käivitamine
```
python -m app.main
```

Seadista enne käivitamist juurkausta `.env` failis `SPREADSHEET_ID`,
`CREDENTIALS_FILE` ja `STREAM_URL`. Tuvastused lisatakse Google Sheetsi
`Test_sheet` lehele veergudesse aeg, numbrimärk ja OCR-kindlus.

Sama numbrimärki ei lisata uuesti 60 sekundi jooksul. Võrdlemisel ei arvestata
tähesuurust, tühikuid ega kirjavahemärke. Esmakordsel käivitamisel luuakse
Google API jaoks `token.json` fail.
