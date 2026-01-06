import pickle
from pathlib import Path

from src.config import MODEL_DIR, PROCESSED_DIR

# --------------------------------------------------
# Artifact existence tests
# --------------------------------------------------

def test_processed_data_exists():
    """
    Verify that at least one processed dataset exists.
    """
    processed_files = list(Path(PROCESSED_DIR).glob("cleaned_v0_*.csv"))

    assert len(processed_files) > 0, "No processed data files found"


def test_model_artifact_exists():
    """
    Verify that at least one trained model artifact exists.
    """
    model_files = list(Path(MODEL_DIR).glob("random_forest_best_*.pkl"))

    assert len(model_files) > 0, "No trained model artifacts found"


def test_metadata_exists():
    """
    Verify that metadata file exists for trained models.
    """
    metadata_files = list(Path(MODEL_DIR).glob("metadata_*.txt"))

    assert len(metadata_files) > 0, "No metadata files found"


# --------------------------------------------------
# Artifact validity tests
# --------------------------------------------------

def test_model_pickle_loadable():
    """
    Verify that the model pickle can be loaded successfully.
    """
    model_file = sorted(Path(MODEL_DIR).glob("random_forest_best_*.pkl"))[-1]

    with open(model_file, "rb") as f:
        model = pickle.load(f)

    assert model is not None, "Loaded model is None"


def test_metadata_contains_required_fields():
    """
    Verify metadata file contains required fields.
    """
    metadata_file = sorted(Path(MODEL_DIR).glob("metadata_*.txt"))[-1]

    with open(metadata_file, "r", encoding="utf-8") as f:
        content = f.read()

    required_fields = [
        "ROC_AUC",
        "Timestamp",
        "Training Data"
    ]

    for field in required_fields:
        assert field in content, f"Missing field in metadata: {field}"
