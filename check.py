import pandas as pd
import os
import argparse
from tqdm import tqdm
from collections import Counter

def check_missing_wavs(csv_path):
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Check each wav file and count labels
    total_files = len(df)
    missing_files = 0
    missing_paths = []
    labels = []
    
    for wav_path in tqdm(df['path']):
        # Check if file exists
        if not os.path.exists(wav_path):
            missing_files += 1
            missing_paths.append(wav_path)
        
        # Extract label from path
        try:
            label = wav_path.split('/')[-3]
            labels.append(label)
        except IndexError:
            print(f"Warning: Couldn't extract label from path: {wav_path}")
    
    # Calculate missing ratio
    missing_ratio = missing_files / total_files
    print(f"\nTotal files: {total_files}")
    print(f"Missing files: {missing_files}")
    print(f"Missing ratio: {missing_ratio:.2%}")
    
    if missing_paths:
        print("\nFirst few missing files:")
        for path in missing_paths[:5]:
            print(path)
    
    # Count and sort labels
    label_counts = Counter(labels)
    print("\nLabel distribution (sorted by count):")
    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_files) * 100
        print(f"{label}: {count} ({percentage:.2f}%)")

def main():
    parser = argparse.ArgumentParser(description='Check missing WAV files in dataset')
    parser.add_argument('--csv', type=str, required=True, help='Path to CSV file containing wav paths')
    args = parser.parse_args()
    
    check_missing_wavs(args.csv)

if __name__ == "__main__":
    main()
