# ==================================================
# NON-GUI BACKEND (MANDATORY FOR PIPELINES)
# ==================================================
import matplotlib
matplotlib.use("Agg")
import sys
# ==================================================
# IMPORTS
# ==================================================
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import pickle
import traceback

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    ParameterGrid
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay
)

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import PROCESSED_DIR, MODEL_DIR, MLFLOW_URI, EXPERIMENT_NAME, LOGS

# ==================================================
# CONFIGURATION
# ==================================================

TIMESTAMP = datetime.now().strftime("%d_%m_%y_%H_%M")

MODEL_PATH = Path(MODEL_DIR)
PROCESSED_PATH = Path(PROCESSED_DIR)

LOG_PATH = Path(LOGS) / "model"
ERROR_LOG = Path(LOGS) / "errors" / "Errors.log"

MODEL_PATH.mkdir(parents=True, exist_ok=True)
LOG_PATH.mkdir(parents=True, exist_ok=True)
ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)

RUN_LOG_FILE = LOG_PATH / f"log_{TIMESTAMP}.log"

# ==================================================
# LOGGING HELPERS
# ==================================================

def log(text: str):
    with open(RUN_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def log_error(text: str):
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(f"{TIMESTAMP} ---- {text}\n")

# ==================================================
# MLFLOW SETUP (FUTURE-SAFE)
# ==================================================

mlflow.set_tracking_uri(MLFLOW_URI)  # e.g. sqlite:///../mlflow.db
mlflow.set_experiment(EXPERIMENT_NAME)

# ==================================================
# MAIN TRAINING PIPELINE
# ==================================================

try:
    log("=== RANDOM FOREST TRAINING PIPELINE STARTED ===")

    # --------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------
    latest_data_file = sorted(PROCESSED_PATH.glob("cleaned_v0_*.csv"))[-1]
    df = pd.read_csv(latest_data_file)

    X = df.drop(columns=["target"])
    y = df["target"]

    log(f"Loaded data: {latest_data_file.name}")
    log(f"Shape: {df.shape}")

    # --------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # --------------------------------------------------
    # METRIC FUNCTION
    # --------------------------------------------------
    def evaluate(model, X_test, y_test):
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob)
        }

    # --------------------------------------------------
    # HYPERPARAMETER GRID
    # --------------------------------------------------
    rf_param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_roc_auc = -1.0
    best_model = None
    best_params = None
    best_accuracy = None
    best_precision = None
    best_recall = None
    # --------------------------------------------------
    # GRID SEARCH WITH FULL MLFLOW LOGGING
    # --------------------------------------------------
    for params in ParameterGrid(rf_param_grid):

        run_name = (
            f"RF_ne{params['n_estimators']}_"
            f"md{params['max_depth']}_"
            f"ms{params['min_samples_split']}_"
            f"ml{params['min_samples_leaf']}"
        )

        log(f"Starting MLflow run: {run_name}")

        with mlflow.start_run(run_name=run_name):

            model = RandomForestClassifier(
                **params,
                random_state=42,
                n_jobs=-1
            )

            cv_results = cross_validate(
                model,
                X_train,
                y_train,
                cv=cv,
                scoring=["accuracy", "precision", "recall", "roc_auc"]
            )

            mean_metrics = {
                "cv_accuracy": np.mean(cv_results["test_accuracy"]),
                "cv_precision": np.mean(cv_results["test_precision"]),
                "cv_recall": np.mean(cv_results["test_recall"]),
                "cv_roc_auc": np.mean(cv_results["test_roc_auc"]),
            }

            mlflow.log_params(params)
            mlflow.log_metrics(mean_metrics)

            # Train final model
            model.fit(X_train, y_train)

            test_metrics = evaluate(model, X_test, y_test)
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

            # ROC Curve
            RocCurveDisplay.from_estimator(model, X_test, y_test)

            # Log model (MLflow sklearn flavor)
            mlflow.sklearn.log_model(model, name="model")

            # Track best model
            if mean_metrics["cv_roc_auc"] > best_roc_auc:
                best_roc_auc = mean_metrics["cv_roc_auc"]
                best_model = model
                best_params = params
                best_accuracy = mean_metrics["cv_accuracy"]
                best_precision = mean_metrics["cv_precision"]
                best_recall = mean_metrics["cv_recall"]

    # --------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------
    if best_model is None:
        log_error("No valid model trained")

    best_model_path = MODEL_PATH / f"random_forest_best_{TIMESTAMP}.pkl"
    with open(best_model_path, "wb") as f:
        pickle.dump(best_model, f)

    metadata_path = MODEL_PATH / f"metadata_{TIMESTAMP}.txt"
    with open(metadata_path, "w") as f:

        f.write(f"Accuracy: {best_accuracy}\n")
        f.write(f"Precision: {best_precision}\n")
        f.write(f"Recall: {best_recall}\n")
        f.write(f"ROC_AUC: {best_roc_auc}\n")
        f.write(f"max_depth: {best_params.get('max_depth', None)}\n")
        f.write(f"Training Data: {latest_data_file.name}\n")
        f.write(f"Timestamp: {TIMESTAMP}\n")

    log(f"Best model saved at: {best_model_path}")
    log("=== TRAINING PIPELINE COMPLETED SUCCESSFULLY ===")

except Exception as e:
    log_error("Training pipeline failed")
    log_error(str(e))
    log_error(traceback.format_exc())
