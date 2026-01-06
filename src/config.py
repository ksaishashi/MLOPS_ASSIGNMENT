
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RAW_LATEST_DIR = BASE_DIR / "data" / "latestdata"
ARCHIVE_DIR = BASE_DIR / "data" / "archive"

LOGS_ROOT = BASE_DIR / "logs" / "data"
LOGS = BASE_DIR / "logs"

MODEL_DIR = BASE_DIR / "models" / "develop"
MODEL_DIR_PROD = BASE_DIR / "models" / "latest"
MODELS_DIR = BASE_DIR / "models"
DEVELOP_DIR = BASE_DIR / "models" / "develop"
LATEST_PATH = BASE_DIR / "models" / "latest"
ARCHIVE_DIR_model = BASE_DIR / "models" / "archive"

MLFLOW_URI = "sqlite:///mlflow.db"   
EXPERIMENT_NAME = "Mlops_Assignment"

DATASET_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"