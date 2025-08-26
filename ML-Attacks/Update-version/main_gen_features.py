import os
import pandas as pd
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from extract_features import extract_features
from tqdm import tqdm  # for a progress bar (optional, but nice)
from utils_extract_features import load_label_map

def process_file(path, labels):
    """Process each file and return features as a dictionary."""
    if not path.endswith('.pcap'):
        return None
    try:
        row = extract_features(path, labels)
        return row
    
    except Exception as e:
        print(f"Failed on {path}: {e}")
        return None

def get_all_pcap_files(dataset_dir):
    """Recursively get all .pcap files in the dataset directory."""
    pcap_files = []
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.pcap'):
                pcap_files.append(os.path.join(root, file))
    return pcap_files

def main(dataset_dir):
    all_files = get_all_pcap_files(dataset_dir)
    labels = load_label_map()
    results = []

    with tqdm(total=len(all_files), desc="Processing Files", unit="file") as pbar:
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(process_file, path, labels): path for path in all_files}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
                pbar.update(1)

    df = pd.DataFrame(results)
    df.to_csv("feature_full.csv", index=False)
    print("Saved: feature_full")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process a dataset directory of .pcap files.")
    parser.add_argument('dataset_dir', help="Path to the dataset directory")
    args = parser.parse_args()
    main(args.dataset_dir)