from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
import torch
import soundfile as sf
import argparse
from tqdm import tqdm
import csv
import random
import json

# Returns audio data as numpy array and sample rate


def language_identification(audio, sr, processor, model: Wav2Vec2ForSequenceClassification, lang_list):
    """
    Function to identify the language of an audio file from a predefined list of languages.
    
    Args:
        audio: Audio waveform
        sr: Sample rate
        processor: Initialized AutoFeatureExtractor
        model: Initialized Wav2Vec2ForSequenceClassification
        LANG_LIST: List of candidate languages in ISO 639-2/3 format (e.g., ['eng', 'jpn', ...])
    
    Returns:
        detected_lang: Most likely language from LANG_LIST
    """
    
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs).logits
    
    # Get probabilities for all languages
    probs = torch.nn.functional.softmax(outputs, dim=-1)[0]
    
    # Filter probabilities to only include languages in LANG_LIST
    filtered_probs = {}
    for i, label in model.config.id2label.items():
        if label in lang_list:
            filtered_probs[label] = probs[i].item()
    
    # Get the most likely language from the filtered list
    detected_lang = max(filtered_probs.items(), key=lambda x: x[1])[0]
    
    
    return detected_lang


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Language Identification')
    parser.add_argument('--input_csv', type=str, help='Path to audio file')
    parser.add_argument('--lang_list', type=str, default="", help='Path to language list')
    args = parser.parse_args()
    
    model_id = "facebook/mms-lid-4017"

    processor = AutoFeatureExtractor.from_pretrained(model_id)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id)
    
    correct = 0
    total = 0
    
    if args.lang_list != "":
        with open(args.lang_list, 'r', encoding='utf-8') as f:
            languages = json.load(f) 
    
    lang_list = [lang['code'] for lang in languages]
    print("Starting language identification...")
    
    lang_record = {}
    
    with open(args.input_csv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        rows = list(reader)
        random.shuffle(rows)
        for row in tqdm(rows):
            path = row[0]
            label = path.split('/')[-3]
            if lang_record[label][1] >= 50:
                continue
            
            audio, sr = sf.read(path)
            pred_lid = language_identification(audio, sr, processor, model, lang_list)
            
            if label not in lang_record:
                lang_record[label] = [0, 0]
            if pred_lid == label:
                correct += 1
                lang_record[label][0] += 1
            total += 1
            lang_record[label][1] += 1
            
    lang_accuracy = {k: v[0]/v[1] for k, v in lang_record.items()}  
    
    print(f"Accuracy: {correct/total}")
    print("Language-wise accuracy:")
    sorted_lang_accuracy = {k: v for k, v in sorted(lang_accuracy.items(), key=lambda item: item[1], reverse=True)}
    for lang, acc in sorted_lang_accuracy.items():
        print(f"{lang}: {acc}")