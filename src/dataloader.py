import os
import tarfile
import glob
import pandas as pd
from src import config

def extract_data(file_path):
    # Check if the extracted folder already exists in the data directory
    if not os.path.exists(config.EXTRACT_PATH):
        print(f"Extracting {file_path} to {config.DATA_DIR}...")
        with tarfile.open(file_path, "r:gz") as tar:
            # We extract it into the 'data' folder
            tar.extractall(path=config.DATA_DIR)
    else:
        print("Data already extracted.")

def load_to_dataframe(dataset_type='train'):
    data = []
    # Loop through 'pos' and 'neg' folders inside data/aclImdb/
    for label in ['pos', 'neg']:
        # Updated path to include the data directory
        path = os.path.join(config.EXTRACT_PATH, dataset_type, label, '*.txt')
        files = glob.glob(path)
        
        if not files:
            print(f"Warning: No files found at {path}")

        for file_path in files:
            with open(file_path, 'r', encoding='utf-8') as f:
                review = f.read()
                data.append({
                    'text': review,
                    'sentiment': 1 if label == 'pos' else 0
                })
                
    return pd.DataFrame(data)