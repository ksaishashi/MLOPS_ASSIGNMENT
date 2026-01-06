import shutil
from pathlib import Path
from datetime import datetime
import traceback
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from src.config import LOGS, LATEST_PATH, DEVELOP_DIR, ARCHIVE_DIR_model
# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path("..")

LOGS_ROOT = Path(LOGS)
LATEST_DIR = Path(LATEST_PATH)
DEVELOP_PATH = Path(DEVELOP_DIR)
DEPLOY_LOG_DIR = LOGS_ROOT / "deployment"
ERROR_LOG = LOGS_ROOT / "errors" / "Errors.log"
ARCHIVE_PATH = Path(ARCHIVE_DIR_model)
TIMESTAMP = datetime.now().strftime("%d_%m_%y_%H_%M")
RUN_LOG_FILE = DEPLOY_LOG_DIR / f"log_{TIMESTAMP}.log"

# Ensure directories exist
DEPLOY_LOG_DIR.mkdir(parents=True, exist_ok=True)
ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
LATEST_DIR.mkdir(parents=True, exist_ok=True)
ARCHIVE_PATH.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Logging helpers
# --------------------------------------------------

def log(text: str):
    with open(RUN_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def log_error(text: str):
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(f"{TIMESTAMP} ---- {text}\n")

# --------------------------------------------------
# Utility functions
# --------------------------------------------------

def get_latest_metadata(folder: Path):
    files = list(folder.glob("metadata_*.txt"))
    if not files:
        return None
    return sorted(files, key=lambda f: f.stat().st_mtime)[-1]

def parse_metadata(metadata_file: Path) -> dict:
    data = {}
    with open(metadata_file, "r", encoding="utf-8") as f:
        for line in f:
            if ":" in line:
                k, v = line.split(":", 1)
                data[k.strip()] = v.strip()
    return data

def extract_roc_auc(meta: dict) -> float:
    if "ROC_AUC" in meta:
        return float(meta["ROC_AUC"])
    if "Best CV ROC-AUC" in meta:
        return float(meta["Best CV ROC-AUC"])
    raise ValueError("ROC_AUC not found in metadata")

def extract_timestamp(meta: dict) -> str:
    return meta.get("Timestamp")

def archive_latest_model(latest_meta: Path):
    latest_info = parse_metadata(latest_meta)
    ts = extract_timestamp(latest_info)

    latest_model = LATEST_DIR / "model.pkl"

    if latest_model.exists():
        archived_model = ARCHIVE_PATH / f"model_{ts}.pkl"
        shutil.move(str(latest_model), archived_model)
        log(f"Archived model as {archived_model.name}")

    archived_meta = ARCHIVE_PATH / latest_meta.name
    shutil.move(str(latest_meta), archived_meta)
    log(f"Archived metadata as {archived_meta.name}")

def promote_develop_model(develop_meta: Path):
    develop_info = parse_metadata(develop_meta)
    ts = extract_timestamp(develop_info)

    develop_model = next(DEVELOP_PATH.glob("*.pkl"))

    shutil.copy(develop_model, LATEST_DIR / "model.pkl")
    shutil.copy(develop_meta, LATEST_DIR / develop_meta.name)

    log(f"Promoted model from develop → latest (timestamp {ts})")

# --------------------------------------------------
# MAIN DEPLOYMENT PIPELINE
# --------------------------------------------------

def main():
    try:
        log("=== MODEL DEPLOYMENT PIPELINE STARTED ===")

        develop_meta = get_latest_metadata(DEVELOP_PATH)
        latest_meta = get_latest_metadata(LATEST_DIR)

        if develop_meta is None:
            log("No metadata found in develop folder. Deployment skipped.")
            return

        develop_info = parse_metadata(develop_meta)
        develop_roc = extract_roc_auc(develop_info)

        log(f"Develop ROC_AUC: {develop_roc}")

        # --------------------------------------------------
        # First deployment
        # --------------------------------------------------
        if latest_meta is None:
            log("No existing deployed model. Performing first deployment.")
            promote_develop_model(develop_meta)
            log("=== DEPLOYMENT COMPLETED (FIRST DEPLOY) ===")
            return

        latest_info = parse_metadata(latest_meta)
        latest_roc = extract_roc_auc(latest_info)

        log(f"Latest ROC_AUC: {latest_roc}")

        # --------------------------------------------------
        # Blue-Green decision
        # --------------------------------------------------
        if develop_roc > latest_roc:
            log("New model outperforms deployed model.")

            archive_latest_model(latest_meta)
            promote_develop_model(develop_meta)

            log("Deployment successful. New model is now active.")
        else:
            log("New model does NOT outperform deployed model.")
            log("Deployment skipped. Existing model retained.")

        log("=== MODEL DEPLOYMENT PIPELINE COMPLETED ===")

    except Exception as e:
        log_error("Deployment pipeline failed")
        log_error(str(e))
        log_error(traceback.format_exc())

# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    main()
