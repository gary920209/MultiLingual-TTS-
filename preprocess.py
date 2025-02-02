import csv
import re
import unicodedata
import argparse

def remove_punctuation(sentence):
    new_sentence = ""
    for char in sentence:
        # all unicode punctuation is of type P
        if unicodedata.category(char).startswith('P'):
            continue
        else:
            new_sentence = f"{new_sentence}{char}"
    return new_sentence

def normalize_text(text, lang):
    # Determine if we should remove spaces based on language
    remove_spaces = lang in ['cmn', 'jpn', 'tha']  # Chinese, Japanese, Thai
    
    # Remove spaces if needed
    if remove_spaces:
        text = re.sub(r"\s", "", text)
    
    # Remove punctuation
    text = remove_punctuation(text)
    
    # Convert to uppercase
    text = text.upper()
    
    return text

def process_line(line):
    # First split by tab
    tab_parts = line.strip().split('\t')
    
    # Then split the first part by space if it contains spaces
    all_parts = []
    for part in tab_parts:
        all_parts.extend(part.split())
    
    # If we don't have at least 3 parts, something is wrong
    if len(all_parts) < 3:
        return None
    
    # First part is always ID
    id_part = all_parts[0]
    
    # Everything from third part onwards is transcript (join with space)
    text = ' '.join(all_parts[2:])
    
    # Extract language code
    lang = id_part.split('_')[1]
    
    # Normalize the text according to CER rules
    normalized_text = normalize_text(text, lang)
    
    # Base path from template
    base_path = "/work/dlhlpgp7/mlsuperb2_challenge/data"
    
    # Construct the new path
    if id_part.startswith('cv_'):
        path = f"{base_path}/commonvoice/{lang}/wav/{id_part}.wav"
    else:
        dataset = id_part.split('_')[0]
        path = f"{base_path}/{dataset}/{lang}/wav/{id_part}.wav"
    
    return [path, normalized_text]


# Parse command line arguments
parser = argparse.ArgumentParser(description='Preprocess text and generate CSV')
parser.add_argument('-i', '--input_file', type=str, help='Path to the input file')
parser.add_argument('-o', '--output_file', type=str, help='Path to the output CSV file')
args = parser.parse_args()


# Read input from file
with open(args.input_file, 'r', encoding='utf-8') as f:
    input_lines = f.readlines()

# Process the input and write to CSV
with open(args.output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['path', 'text'])
    
    for line in input_lines:
        result = process_line(line)
        if result:
            writer.writerow(result)