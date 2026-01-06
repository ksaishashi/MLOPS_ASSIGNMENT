import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

# --------------------------------------------------
# Config & Paths
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import PROCESSED_DIR, RAW_LATEST_DIR, LOGS_ROOT, ARCHIVE_DIR

PROCESSED_PATH = Path(PROCESSED_DIR)
RAW_LATEST_PATH = Path(RAW_LATEST_DIR)
ARCHIVE_PATH = Path(ARCHIVE_DIR)

TIMESTAMP = datetime.now().strftime("%d_%m_%y_%H_%M")
RUN_LOG_DIR = Path(LOGS_ROOT) / f"data_{TIMESTAMP}"
RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)

ERROR_LOG = Path(LOGS_ROOT) / "errors/Errors.log"

sns.set(style="whitegrid")
plt.style.use("seaborn-v0_8")

# --------------------------------------------------
# Utility functions
# --------------------------------------------------

def log_text(text: str, filename="data_report.txt"):
    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUN_LOG_DIR / filename, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def log_error(text: str):
    ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(f"{TIMESTAMP} ---- {text}\n")

def save_plot(fig, name):
    fig.savefig(RUN_LOG_DIR / name, bbox_inches="tight")
    plt.close(fig)

def archive_file_with_timestamp(file_path: Path, archive_dir: Path):
    archive_dir.mkdir(parents=True, exist_ok=True)
    new_name = f"{file_path.stem}_{TIMESTAMP}{file_path.suffix}"
    shutil.move(str(file_path), archive_dir / new_name)

# --------------------------------------------------
# File selection helpers
# --------------------------------------------------

def get_latest_csv_file_safe(folder: Path) -> Path | None:
    try:
        files = list(folder.glob("*.csv"))
        if not files:
            log_error(f"No CSV files found in {folder}")
            return None
        return max(files, key=lambda x: x.stat().st_mtime)
    except Exception as e:
        log_error(f"Failed to read CSV from {folder}: {str(e)}")
        return None

def get_latest_processed_file_by_name(folder: Path) -> Path | None:
    files = list(folder.glob("cleaned_v0_*.csv"))
    if not files:
        return None

    def extract_ts(f: Path):
        return datetime.strptime(
            f.stem.replace("cleaned_v0_", ""),
            "%d_%m_%y_%H_%M"
        )

    return sorted(files, key=extract_ts)[-1]

# --------------------------------------------------
# Data loading
# --------------------------------------------------
def load_raw_data() -> pd.DataFrame:
    file_path = get_latest_csv_file_safe(RAW_LATEST_PATH)
    if file_path is None:
        return pd.DataFrame()

    columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak",
        "slope", "ca", "thal", "target"
    ]

    try:
        # First read: let pandas infer header
        df = pd.read_csv(file_path)

        # If header is missing or wrong, fix it
        if list(df.columns) != columns:
            df = pd.read_csv(file_path, header=None, names=columns)

        log_text(f"Loaded raw file: {file_path.name}")
        archive_file_with_timestamp(file_path, ARCHIVE_PATH)
        return df

    except Exception as e:
        log_error(f"Failed to load CSV {file_path.name}: {str(e)}")
        return pd.DataFrame()

# --------------------------------------------------
# Cleaning & preprocessing
# --------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace("?", np.nan)
    df = df.apply(pd.to_numeric)

    df["ca"].fillna(df["ca"].median(), inplace=True)
    df["thal"].fillna(df["thal"].mode()[0], inplace=True)

    df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

    # df["age_bin"] = pd.cut(
    #     df["age"],
    #     bins=range(0, int(df["age"].max()) + 5, 5),
    #     right=False
    # )
    # df["age_bin"] = df["age_bin"].astype("category").cat.codes
    df = pd.get_dummies(
        df,
        columns=["cp", "restecg", "thal"],
        drop_first=True
    )

    cols = [c for c in df.columns if c != "target"] + ["target"]
    return df[cols]

# --------------------------------------------------
# Append existing processed data
# --------------------------------------------------

def append_existing_data(df_new: pd.DataFrame) -> pd.DataFrame:
    latest_processed = get_latest_processed_file_by_name(PROCESSED_PATH)
    if latest_processed is None:
        log_text("No existing processed data found. Using new data only.")
        return df_new

    df_old = pd.read_csv(latest_processed)
    # df_old["age_bin"] = df_old["age_bin"].astype("category").cat.codes
    # if "age_bin" in df_old.columns:
    #     if not is_categorical_dtype(df_old["age_bin"]):
    #         df_old["age_bin"] = df_old["age_bin"].astype("category").cat.codes
    log_text(f"Appending with processed data: {latest_processed.name}")
    return pd.concat([df_old, df_new], ignore_index=True)

# --------------------------------------------------
# EDA & Logging
# --------------------------------------------------

def run_eda(df: pd.DataFrame):
    log_text("\n--- DATA SUMMARY ---")
    log_text(f"Shape: {df.shape}")
    log_text(f"\nMissing values:\n{df.isna().sum()}")
    log_text("\nTarget distribution:")
    log_text(str(df["target"].value_counts()))
    log_text("\nDescribe:")
    log_text(str(df.describe()))

    fig = df.hist(figsize=(16, 12), bins=20)
    plt.suptitle("Feature Distributions")
    save_plot(plt.gcf(), "histograms.png")

    df_corr = df.copy()
    # if "age_bin" in df_corr:
    #     df_corr["age_bin"] = df_corr["age_bin"].astype("category").cat.codes
    df_corr[df_corr.select_dtypes("bool").columns] = \
        df_corr.select_dtypes("bool").astype(int)

    corr = df_corr.corr()

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(corr, cmap="coolwarm", linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap")
    save_plot(fig, "correlation_heatmap.png")

    target_corr = corr["target"].drop("target").sort_values(ascending=False)
    log_text("\nTarget correlations:")
    log_text(str(target_corr))

    fig, ax = plt.subplots()
    sns.countplot(x="target", data=df, ax=ax)
    ax.set_title("Class Distribution")
    save_plot(fig, "class_balance.png")

    fig, ax = plt.subplots()
    sns.boxplot(x="target", y="thalach", data=df, ax=ax)
    ax.set_title("Thalach vs Target")
    save_plot(fig, "thalach_vs_target.png")

# --------------------------------------------------
# Save processed data
# --------------------------------------------------

def save_processed_data(df: pd.DataFrame):
    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    output_file = PROCESSED_PATH / f"cleaned_v0_{TIMESTAMP}.csv"
    df.to_csv(output_file, index=False)
    log_text(f"Saved processed dataset: {output_file.name}")

# --------------------------------------------------
# Main pipeline
# --------------------------------------------------

def main():
    try:
        log_text("=== PREPROCESSING PIPELINE STARTED ===")

        df_raw = load_raw_data()

        if len(df_raw) <= 10:
            log_text(
                f"Data quality check failed: only {len(df_raw)} rows found "
                "(minimum required: 10). Pipeline stopped."
            )
            log_text("=== PIPELINE COMPLETED (NO DATA) ===")
            return

        df_clean = clean_data(df_raw)
        df_final = append_existing_data(df_clean)

        run_eda(df_final)
        save_processed_data(df_final)

        log_text("=== PIPELINE COMPLETED SUCCESSFULLY ===")
    except Exception as e:
        log_error(f"Error in preprocessing pipeline with {e}")

if __name__ == "__main__":
    main()
