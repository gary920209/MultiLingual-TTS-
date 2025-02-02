import pandas as pd
import os
import argparse
from tqdm import tqdm

def check_missing_wavs(csv_path):
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Check each wav file
    total_files = len(df)
    missing_files = 0
    missing_paths = []
    
    for wav_path in tqdm(df['path']):
        if not os.path.exists(wav_path):
            missing_files += 1
            missing_paths.append(wav_path)
    
    missing_ratio = missing_files / total_files
    print(f"Total files: {total_files}")
    print(f"Missing files: {missing_files}")
    print(f"Missing ratio: {missing_ratio:.2%}")
    
    if missing_paths:
        print("\nFirst few missing files:")
        for path in missing_paths[:5]:
            print(path)

def main():
    parser = argparse.ArgumentParser(description='Check missing WAV files in dataset')
    parser.add_argument('--csv', type=str, required=True, help='Path to CSV file containing wav paths')
    args = parser.parse_args()
    
    check_missing_wavs(args.csv)

if __name__ == "__main__":
    main()
