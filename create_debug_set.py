import pandas as pd
from pathlib import Path
import argparse

def create_debug_set(input_csv, output_csv, sample):
    # Read the CSV file
    df = pd.read_csv(input_csv)

    # Extract language codes from path
    df['lang'] = df['path'].apply(lambda x: Path(x).parts[-3])

    # Get 10 example per language, if there is less than 20, take all (randomly)
    debug_df = df.groupby('lang').sample(n=sample, random_state=42).reset_index(drop=True)

    # Drop the language column as it's not in original format
    debug_df = debug_df[['path', 'text']]

    # Save to new CSV
    debug_df.to_csv(output_csv, index=False)
    print(f"Created debug dataset with {len(debug_df)} examples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str)
    parser.add_argument("--output_csv", type=str)
    parser.add_argument("--sample", type=int, default=20)
    args = parser.parse_args()
    input_csv = args.input_csv  # Replace with your input CSV path
    output_csv = args.output_csv  # Output CSV name
    sample = args.sample  # Number of samples per language
    create_debug_set(input_csv, output_csv, sample)
