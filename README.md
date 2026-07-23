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

Esmakordsel käivitamisel luuakse Google API jaoks token.json fail.