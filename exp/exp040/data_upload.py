import argparse
from kaggle.api.kaggle_api_extended import KaggleApi
import json
import os

def dataset_create_new(dataset_name: str, upload_dir: str):
    dataset_name = dataset_name.replace("_", "-").replace(".", "-")
    
    dataset_metadata = {}
    dataset_metadata["id"] = f"sinchir0/{dataset_name}"
    dataset_metadata["licenses"] = [{"name": "CC0-1.0"}]
    dataset_metadata["title"] = dataset_name
    
    with open(os.path.join(upload_dir, "dataset-metadata.json"), "w") as f:
        json.dump(dataset_metadata, f, indent=4)
    
    api = KaggleApi()
    api.authenticate()
    api.dataset_create_new(folder=upload_dir, convert_to_csv=False, dir_mode="tar")

if __name__ == "__main__":
    # Pathの指定
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True, help='ディレクトリのパス')
    parser.add_argument('--dataset-name', type=str, required=True, help='データセット名')
    args = parser.parse_args()
    UPLOAD_PATH = args.dir
    DATASET_NAME = args.dataset_name

    print(f"Create Dataset name:{DATASET_NAME}, output_dir:{UPLOAD_PATH}")
    dataset_create_new(dataset_name=DATASET_NAME, upload_dir=UPLOAD_PATH)