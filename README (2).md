# 🎬 YouTube Sentiment Analysis — End‑to‑End MLOps

![Python](https://img.shields.io/badge/Python-3.11%20recommended-blue)
![Flask](https://img.shields.io/badge/Flask-API-informational)
![LightGBM](https://img.shields.io/badge/LightGBM-4.5.0-success)
![MLflow](https://img.shields.io/badge/MLflow-2.17.0%20%28skinny%20ok%29-ff69b4)
![DVC](https://img.shields.io/badge/DVC-3.53.0-9cf)
![AWS%20S3](https://img.shields.io/badge/Storage-AWS%20S3-orange)
![NLTK](https://img.shields.io/badge/NLP-NLTK-yellowgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

An end‑to‑end **MLOps** project that analyzes **YouTube comments sentiment**. It includes data ingestion, text preprocessing, model training/evaluation, experiment tracking (**MLflow**), data/model versioning (**DVC**), and a small serving layer (Flask/Streamlit).

> Your exact folders may differ from this README. All commands below are Windows‑friendly and work from the project root.

---

## 📦 Highlights
- **NLP pipeline** with NLTK (tokenize, stopwords, lemmatize) + visualizations (wordcloud, seaborn).
- **Models**: scikit‑learn & LightGBM with metrics (Accuracy/F1/ROC‑AUC).
- **Tracking**: MLflow runs/params/metrics/artifacts (use `mlflow-skinny` to avoid heavy deps).
- **Versioning**: DVC for datasets & model artifacts; optional **S3** remote.
- **Serving**: Minimal Flask API or Streamlit app for quick local scoring.

---

## 📂 Typical Structure
```
.
├─ requirements.txt
├─ README.md
├─ data/                 # (optional) local data cache
├─ models/               # trained models, vectorizers, label encoders
├─ notebooks/            # EDA / experiments
├─ src/
│  ├─ data/              # ingestion / cleaning
│  ├─ features/          # text preprocessing / vectorization
│  ├─ models/            # train / evaluate scripts
│  ├─ serve/             # Flask/FastAPI/Streamlit app
│  └─ utils/             # helpers
├─ dvc.yaml              # (if present) pipeline stages
├─ params.yaml           # (if present) hyperparameters, paths
└─ MLproject             # (optional) MLflow Projects
```
> If `dvc.yaml` or `MLproject` aren’t present in your copy, skip those sections.

---

## ⚙️ Prerequisites
- **Python 3.11** (recommended). Python **3.13** also works if you keep **mlflow‑skinny** and remove full **mlflow**.
- **Windows PowerShell** (commands below use it)
- (Optional) **AWS CLI** and an S3 bucket if you want remote storage for DVC.

---

## 🚀 Quickstart (Local)

> Run all commands **from the project root**.

### Option A — Recommended (Python 3.11)
```powershell
# Create & activate venv with Python 3.11
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Download NLTK data used by the pipeline
python - << 'PY'
import nltk
for pkg in ["punkt", "stopwords", "wordnet", "omw-1.4", "vader_lexicon"]:
    try:
        nltk.download(pkg)
    except Exception as e:
        print(f"NLTK download failed for {pkg}: {e}")
PY
```

### Option B — Stay on Python 3.13 (avoid pyarrow build)
If `pip install` fails due to **pyarrow** (pulled by `mlflow`), keep **mlflow‑skinny** only:
```powershell
# In case requirements.txt pins 'mlflow==2.17.0', remove that line (keep mlflow-skinny)
(Get-Content requirements.txt) -notmatch '^mlflow==' | Set-Content requirements.txt

python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

## 🧪 Train & Evaluate

### A) DVC pipeline (if `dvc.yaml` exists)
```powershell
dvc init

# Optional: configure S3 remote (needs AWS credentials)
dvc remote add -d s3remote s3://YOUR_BUCKET/YOUR_PREFIX

# Reproduce pipeline (runs ingestion → preprocess → train → evaluate)
dvc repro

# Push artifacts (data/models) to remote
dvc push
```

### B) Plain scripts
If you don’t have `dvc.yaml`, find the training entry point:
```powershell
# List all python files
Get-ChildItem -Recurse -Filter *.py | Select-Object FullName

# Find files with a main() or train code
Get-ChildItem -Recurse -Filter *.py | Select-String -Pattern 'if __name__ == "__main__"' | Select-Object Path, LineNumber, Line

# Run the script that performs training, e.g.:
python src\models\train.py   # adjust path/filename if different
```

### MLflow tracking UI
```powershell
# Local file-based tracking
$env:MLFLOW_TRACKING_URI="file:./mlruns"
mlflow ui --host 0.0.0.0 --port 5001
# Open http://localhost:5001
```

---

## 🔌 Serving (score new text)

This project includes **Flask** in `requirements.txt`. Detect the app file and run it:

```powershell
# Find a Flask entry (a file that imports/creates Flask)
Get-ChildItem -Recurse -Filter *.py | Select-String -Pattern 'from flask|Flask\(' | Select-Object Path -Unique

# Start it (replace path below with the file found above)
python src\serve\app.py

# Open http://localhost:5000
```

**Streamlit** alternative (if repository uses it):
```powershell
streamlit run path\to\streamlit_app.py
# http://localhost:8501
```

**FastAPI** alternative (if `app = FastAPI()` exists):
```powershell
uvicorn path.to.module:app --reload --port 8000
# http://localhost:8000 (docs at /docs)
```

---

## ☁️ DVC + S3 (optional)
```powershell
# AWS credentials (PowerShell env vars)
$env:AWS_ACCESS_KEY_ID="YOUR_KEY_ID"
$env:AWS_SECRET_ACCESS_KEY="YOUR_SECRET"
$env:AWS_DEFAULT_REGION="ap-south-1"

dvc remote add -d s3remote s3://YOUR_BUCKET/YOUR_PREFIX

# Push or pull data/models
dvc push
dvc pull
```

---

## 🔧 Configuration
- **YouTube API key** (only if your ingestion fetches live comments)
  ```powershell
  $env:YOUTUBE_API_KEY="YOUR_API_KEY"
  ```
- **.env** support: if `python-dotenv` is used, put keys in a `.env` file.
- **NLTK**: if runtime errors say a resource is missing, run:
  ```powershell
  python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')"
  ```

---

## 🧩 Example Flask endpoint (sketch)
```python
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("models/model.joblib")
vectorizer = joblib.load("models/vectorizer.joblib")

@app.post("/predict")
def predict():
    text = request.get_json().get("text", "")
    X = vectorizer.transform([text])
    proba = model.predict_proba(X)[0, 1]
    return jsonify({"label": int(proba >= 0.5), "probability": float(proba)})
```

Run:
```powershell
python src\serve\app.py
# POST to http://localhost:5000/predict with {"text": "I love this video!"}
```

---

## 🧭 Troubleshooting
- **`pyarrow` build error on Python 3.13** → remove full `mlflow` (keep `mlflow‑skinny`) or use Python **3.11**.
- **Venv not activating** → once per machine:  
  `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
- **Framework not found** → install missing package or run the correct entry file.
- **Port in use** → change port: `flask run -p 5050` / `uvicorn ... --port 5050`.
- **NLTK resource not found** → download the missing datasets as shown above.

---

## 🗺 Roadmap
- [ ] Add tests & CI (GitHub Actions)
- [ ] Dockerfile for API inference
- [ ] Parameterize with `params.yaml` + `dvc.yaml`
- [ ] SHAP explanations & model cards

---

## 📜 License
MIT (see `LICENSE`).
