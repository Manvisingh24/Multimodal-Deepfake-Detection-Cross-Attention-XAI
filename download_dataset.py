import os
import sys
import json
from pathlib import Path

# Setup directories
ROOT_DIR = Path(__file__).resolve().parent
DATA_RAW = ROOT_DIR / "data" / "raw"
KAGGLE_JSON_PATH = ROOT_DIR / "kaggle.json"

def setup_kaggle_credentials():
    """Configure Kaggle API environment variables."""
    if not KAGGLE_JSON_PATH.exists():
        print("❌ Error: kaggle.json not found in the root directory!")
        print("   Please place kaggle.json in your project root folder.")
        sys.exit(1)
    
    # Read API credentials
    with open(KAGGLE_JSON_PATH, "r") as f:
        creds = json.load(f)
        os.environ["KAGGLE_USERNAME"] = creds.get("username", "")
        os.environ["KAGGLE_KEY"] = creds.get("key", "")
    
    print("✅ Kaggle API credentials configured successfully.")

def download_and_extract_dataset(dataset_identifier, extract_to_folder):
    """Download dataset using Kaggle API and extract contents."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    target_dir = DATA_RAW / extract_to_folder
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📥 Fetching dataset: '{dataset_identifier}'...")
    api.dataset_download_files(dataset_identifier, path=target_dir, unzip=True)
    print(f"✅ Downloaded and extracted to: {target_dir}")

if __name__ == "__main__":
    setup_kaggle_credentials()
    
    # Public Kaggle Deepfake Sample Dataset
    DATASET_HANDLE = "xhlulu/140k-real-and-fake-faces"
    
    try:
        download_and_extract_dataset(DATASET_HANDLE, "faces_dataset")
        print("\n🎉 Phase 2 Dataset Setup Complete!")
    except Exception as e:
        print(f"\n❌ Download failed: {str(e)}")