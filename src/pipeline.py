import subprocess
import sys
from pathlib import Path
from datetime import datetime

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR.parent / "logs" / "pipeline"
ERROR_LOG = BASE_DIR.parent / "logs" / "errors" / "Errors.log"

TIMESTAMP = datetime.now().strftime("%d_%m_%y_%H_%M")
PIPELINE_LOG = LOGS_DIR / f"pipeline_{TIMESTAMP}.log"

LOGS_DIR.mkdir(parents=True, exist_ok=True)
ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Logging helpers
# --------------------------------------------------

def log(msg: str):
    with open(PIPELINE_LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg)

def log_error(msg: str):
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(f"{TIMESTAMP} ---- {msg}\n")
    print(msg, file=sys.stderr)

# --------------------------------------------------
# Runner utility
# --------------------------------------------------

def run_step(script_name: str, step_name: str):
    try:
        log(f"\n=== STARTING: {step_name} ===")

        script_path = BASE_DIR / script_name

        if not script_path.exists():
            raise FileNotFoundError(f"{script_name} not found")

        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            capture_output=True,
            text=True
        )

        log(result.stdout)
        log(f"=== COMPLETED: {step_name} ===")

    except subprocess.CalledProcessError as e:
        log_error(f"{step_name} FAILED")
        log_error(e.stderr)
        raise

    except Exception as e:
        log_error(f"{step_name} FAILED: {str(e)}")
        raise

# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def main():
    try:
        log("############################################")
        log("### END-TO-END MLOPS PIPELINE STARTED ###")
        log(f"Timestamp: {TIMESTAMP}")
        log("############################################")

        run_step("preprocessing_pipeline.py", "DATA PREPROCESSING")
        run_step("training_pipeline.py", "MODEL TRAINING")
        run_step("deploy_pipeline.py", "MODEL DEPLOYMENT")

        log("############################################")
        log("### PIPELINE COMPLETED SUCCESSFULLY ###")
        log("############################################")

    except Exception:
        log_error("PIPELINE EXECUTION STOPPED DUE TO ERROR")

# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    main()
