# Diverzify Freight — Quick Start (FastAPI + Streamlit)

**Requirements:** Python 3.11+ on PATH

## Setup
**Windows (PowerShell)**
```powershell
cd <project>
py -3.11 -m venv .venv; .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**macOS/Linux**
```bash
cd <project>
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run
**Backend (FastAPI)**
```bash
python -m uvicorn app.api.main:app --reload
# http://127.0.0.1:8000/docs
```

**Frontend (Streamlit)** (new terminal, activate venv again)
```bash
streamlit run app/streamlit/app.py
# http://localhost:8501
```

## Troubleshooting
- Activation blocked (Windows): `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned -Force`
- `uvicorn` not found → use `python -m uvicorn ...`
- Wrong interpreter → activate `.venv` first
