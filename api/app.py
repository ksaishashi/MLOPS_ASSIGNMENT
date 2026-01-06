# ==================================================
# FASTAPI MODEL SERVING + MONITORING + RETRAIN
# ==================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import pickle
import subprocess
import sys
import traceback
import time
from threading import Lock
from fastapi.middleware.cors import CORSMiddleware

# --------------------------------------------------
# Project path setup
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import LATEST_PATH, LOGS

# --------------------------------------------------
# Paths & logging
# --------------------------------------------------

MODEL_PATH = Path(LATEST_PATH) / "model.pkl"
LATEST_DIR = Path(LATEST_PATH)

API_LOG_DIR = Path(LOGS) / "api"
ERROR_LOG_DIR = Path(LOGS) / "errors"

API_LOG_DIR.mkdir(parents=True, exist_ok=True)
ERROR_LOG_DIR.mkdir(parents=True, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%d_%m_%y_%H_%M")
API_LOG_FILE = API_LOG_DIR / f"api_{TIMESTAMP}.log"
ERROR_LOG_FILE = ERROR_LOG_DIR / "Errors.log"

# --------------------------------------------------
# In-memory monitoring state
# --------------------------------------------------

START_TIME = time.time()

TOTAL_REQUESTS = 0
SUCCESS_REQUESTS = 0
FAILED_REQUESTS = 0

LAST_PREDICTION_TIME = None
LAST_RETRAIN_TIME = None

REQUEST_LATENCIES = []               # seconds
PREDICTION_COUNTS = {0: 0, 1: 0}
RETRAIN_COUNT = 0

# --------------------------------------------------
# Logging helpers
# --------------------------------------------------

def log_api(msg: str):
    with open(API_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now()} ---- {msg}\n")

def log_error(msg: str):
    with open(ERROR_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now()} ---- {msg}\n")

# --------------------------------------------------
# Load model
# --------------------------------------------------

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    log_api("Model loaded successfully")
except Exception as e:
    log_error("Failed to load model")
    log_error(str(e))
    raise RuntimeError("Model loading failed")

# --------------------------------------------------
# FastAPI app
# --------------------------------------------------

app = FastAPI(
    title="Heart Disease Prediction API",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Allow all origins (safe for assignment/demo)
    allow_credentials=True,
    allow_methods=["*"],        # GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],        # Authorization, Content-Type, etc.
)

# --------------------------------------------------
# Input schema
# --------------------------------------------------
STATE_LOCK = Lock()

class HeartInput(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

# --------------------------------------------------
# Preprocessing (MUST MATCH TRAINING)
# --------------------------------------------------

EXPECTED_FEATURES = [
    "age", "sex", "trestbps", "chol", "fbs", "thalach",
    "exang", "oldpeak", "slope", "ca",
    "cp_2.0", "cp_3.0", "cp_4.0",
    "restecg_1.0", "restecg_2.0",
    "thal_6.0", "thal_7.0"
]

def preprocess_input(data: HeartInput) -> pd.DataFrame:
    df = pd.DataFrame([data.dict()])

    df = df.replace("?", np.nan)
    df = df.apply(pd.to_numeric)

    df["ca"].fillna(df["ca"].median(), inplace=True)
    df["thal"].fillna(df["thal"].mode()[0], inplace=True)

    df = pd.get_dummies(
        df,
        columns=["cp", "restecg", "thal"],
        drop_first=True
    )

    # Ensure all expected columns exist
    for col in EXPECTED_FEATURES:
        if col not in df.columns:
            df[col] = 0

    return df[EXPECTED_FEATURES]

# --------------------------------------------------
# Metadata helpers (for monitoring)
# --------------------------------------------------

def parse_metadata(metadata_file: Path) -> dict:
    data = {}
    with open(metadata_file, "r", encoding="utf-8") as f:
        for line in f:
            if ":" in line:
                k, v = line.split(":", 1)
                data[k.strip()] = v.strip()
    return data

def get_latest_metadata(folder: Path):
    files = list(folder.glob("metadata_*.txt"))
    if not files:
        return None
    return sorted(files, key=lambda f: f.stat().st_mtime)[-1]

def extract_roc_auc(meta: dict) -> float:
    if "ROC_AUC" in meta:
        return float(meta["ROC_AUC"])
    if "Best CV ROC-AUC" in meta:
        return float(meta["Best CV ROC-AUC"])
    return None

# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(input_data: HeartInput):
    global TOTAL_REQUESTS, SUCCESS_REQUESTS, FAILED_REQUESTS
    global LAST_PREDICTION_TIME
    with STATE_LOCK:
        TOTAL_REQUESTS += 1
    start_time = time.time()

    try:
        log_api(f"Request received: {input_data.dict()}")

        X = preprocess_input(input_data)

        pred = int(model.predict(X)[0])
        prob = float(model.predict_proba(X)[0][1])

        latency = time.time() - start_time
        with STATE_LOCK:
            PREDICTION_COUNTS[pred] += 1
            REQUEST_LATENCIES.append(latency)
            SUCCESS_REQUESTS += 1

        LAST_PREDICTION_TIME = datetime.now().isoformat()

        response = {
            "prediction": pred,
            "confidence": round(prob, 4)
        }

        log_api(f"Prediction response: {response}")
        return response

    except Exception as e:
        FAILED_REQUESTS += 1
        log_error("Prediction failed")
        log_error(str(e))
        log_error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Prediction failed")

# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "UP",
        "model_loaded": MODEL_PATH.exists(),
        "uptime_seconds": int(time.time() - START_TIME)
    }

# --------------------------------------------------
# Monitoring endpoint
# --------------------------------------------------

@app.get("/metrics")
def metrics():
    uptime = int(time.time() - START_TIME)

    avg_latency = (
        round(np.mean(REQUEST_LATENCIES) * 1000, 2)
        if REQUEST_LATENCIES else None
    )

    p95_latency = (
        round(np.percentile(REQUEST_LATENCIES, 95) * 1000, 2)
        if len(REQUEST_LATENCIES) >= 5 else None
    )

    rpm = round(
        TOTAL_REQUESTS / (uptime / 60), 2
    ) if uptime > 0 else 0

    latest_meta = get_latest_metadata(LATEST_DIR)
    model_roc = None
    model_timestamp = None

    if latest_meta:
        meta = parse_metadata(latest_meta)
        model_roc = extract_roc_auc(meta)
        model_timestamp = meta.get("Timestamp")

    return {
        "traffic": {
            "total_requests": TOTAL_REQUESTS,
            "successful_requests": SUCCESS_REQUESTS,
            "failed_requests": FAILED_REQUESTS,
            "requests_per_minute": rpm,
        },
        "latency_ms": {
            "avg": avg_latency,
            "p95": p95_latency,
        },
        "predictions": {
            "distribution": PREDICTION_COUNTS,
            "last_prediction_time": LAST_PREDICTION_TIME,
        },
        "model": {
            "path": str(MODEL_PATH),
            "roc_auc": model_roc,
            "timestamp": model_timestamp,
        },
        "system": {
            "uptime_seconds": uptime,
            "retrain_count": RETRAIN_COUNT,
        }
    }

# --------------------------------------------------
# Retrain endpoint
# --------------------------------------------------

@app.post("/retrain")
def retrain():
    global LAST_RETRAIN_TIME, RETRAIN_COUNT

    try:
        log_api("Retraining triggered via API")

        pipeline_path = PROJECT_ROOT / "src" / "pipeline.py"

        subprocess.run(
            [sys.executable, str(pipeline_path)],
            check=True
        )

        LAST_RETRAIN_TIME = datetime.now().isoformat()
        RETRAIN_COUNT += 1

        log_api("Retraining completed successfully")

        return {
            "status": "SUCCESS",
            "timestamp": LAST_RETRAIN_TIME
        }

    except Exception as e:
        log_error("Retraining failed")
        log_error(str(e))
        log_error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Retraining failed")

# --------------------------------------------------
# Entry point (NO manual uvicorn command needed)
# --------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
