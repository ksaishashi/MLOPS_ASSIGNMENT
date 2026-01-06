import pandas as pd
from pathlib import Path
import pytest

from src.config import PROCESSED_DIR, MODEL_DIR

@pytest.fixture(scope="session")
def processed_df():
    file = sorted(Path(PROCESSED_DIR).glob("cleaned_v0_*.csv"))[-1]
    return pd.read_csv(file)

@pytest.fixture(scope="session")
def trained_model_path():
    return sorted(Path(MODEL_DIR).glob("random_forest_best_*.pkl"))[-1]
