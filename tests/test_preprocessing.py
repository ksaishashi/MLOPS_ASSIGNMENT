from src.preprocessing_pipeline import clean_data
import pandas as pd

RAW_COLUMNS = [
    "age","sex","cp","trestbps","chol","fbs",
    "restecg","thalach","exang","oldpeak",
    "slope","ca","thal","target"
]

EXPECTED_COLUMNS = {
    "age","sex","trestbps","chol","fbs","thalach",
    "exang","oldpeak","slope","ca",
    "cp_2.0","cp_3.0","cp_4.0",
    "restecg_1.0","restecg_2.0",
    "thal_6.0","thal_7.0",
    "target"
}

def test_clean_data_output_schema(processed_df):
    assert EXPECTED_COLUMNS.issubset(set(processed_df.columns))

def test_no_missing_values(processed_df):
    assert processed_df.isna().sum().sum() == 0

def test_target_binary(processed_df):
    assert set(processed_df["target"].unique()).issubset({0, 1})

def test_all_numeric(processed_df):
    assert all(dtype.kind in "iufb" for dtype in processed_df.dtypes)
