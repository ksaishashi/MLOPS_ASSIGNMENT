import os
import requests
import zipfile
from io import BytesIO
from src.config import DATA_DIR, DATASET_URL


def download_and_extract():
    os.makedirs(DATA_DIR, exist_ok=True)

    response = requests.get(DATASET_URL)
    response.raise_for_status()

    with zipfile.ZipFile(BytesIO(response.content)) as z:
        z.extractall(DATA_DIR)

    print(f"Dataset download and saved to: {DATA_DIR}")

if __name__ == "__main__":
    download_and_extract()
