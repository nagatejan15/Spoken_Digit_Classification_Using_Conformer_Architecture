from datasets import load_dataset
import pathlib
import os
from dotenv import load_dotenv
load_dotenv()

dataset_name = "mteb/free-spoken-digit-dataset"
local_save_directory = pathlib.Path(os.getenv('dataset_path'))
local_save_directory.mkdir(exist_ok=True)

def download_dataset(FORCE_DOWNLOAD = False):
    if FORCE_DOWNLOAD or not os.path.exists(local_save_directory) or not os.listdir(local_save_directory):
        dataset_name = "mteb/free-spoken-digit-dataset"

        try:
            dataset = load_dataset(dataset_name)
            for split, data in dataset.items():
                file_path = local_save_directory / f"{split}.parquet"
                data.to_parquet(file_path)

            print(f"Dataset successfully downloaded and saved to '{local_save_directory}'")
            return True, None

        except Exception as e:
            print(f"An error occurred: {e}")
            return False, e
    
    else:
        print("Files already downloaded, skipping download")
        return True, None
