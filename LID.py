from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
import torch
import soundfile as sf
import argparse
from tqdm import tqdm
import csv

# Returns audio data as numpy array and sample rate


def language_identification(audio, sr, processor, model):

    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs).logits

    lang_id = torch.argmax(outputs, dim=-1)[0].item()
    detected_lang = model.config.id2label[lang_id]

    return detected_lang


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Language Identification')
    parser.add_argument('--input_csv', type=str, help='Path to audio file')
    args = parser.parse_args()
    
    model_id = "facebook/mms-lid-4017"

    processor = AutoFeatureExtractor.from_pretrained(model_id)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id)
    
    correct = 0
    total = 0
    
    with open(args.input_csv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            path = row[0]
            audio, sr = sf.read(path)
            lang = language_identification(audio, sr, processor, model)
            label = path.split('/')[-3]
            if lang == label:
                correct += 1
            total += 1
    
    print(f"Accuracy: {correct/total}")