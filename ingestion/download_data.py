import os
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi

load_dotenv()

def download_kaggle_dataset(dataset_slug: str, download_path: str = "data") -> None:
    """
    Downloads and unzips a dataset from Kaggle using the official API.
    
    Args:
        dataset_slug (str): The ID of the dataset (e.g., 'username/dataset-name').
        download_path (str): The directory to save the files.
    """
    # Ensure the download directory exists
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    api = KaggleApi()
    api.authenticate()

    print(f"Downloading {dataset_slug} to {download_path}...")
    try:
        # Download and unzip
        api.dataset_download_files(dataset_slug, path=download_path, unzip=True)
        print(f"Successfully downloaded {dataset_slug}.")
    except OSError as e:
        print(f"Failed to download {dataset_slug} due to OS error: {e}")
    except Exception as e:
        # Kaggle API might raise specific exceptions but we keep generic catch for now if unsure of API behavior
        # But we improve error reporting.
        print(f"Failed to download {dataset_slug} due to an unexpected error: {e}")

if __name__ == "__main__":
    datasets = [
        "madhurpant/srimad-bhagawatam-bhagavata-purana-dataset",
        "a2m2a2n2/bhagwad-gita-dataset"
    ]
    
    for ds in datasets:
        download_kaggle_dataset(ds)
    
    print("Download complete. Verify column names match 'Speaker', 'Text', etc. before running ingest.py.")