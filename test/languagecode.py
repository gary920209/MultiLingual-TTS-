import csv
import re
import json
from datetime import datetime

def load_iso_mapping(mapping_tsv_path):
    """Load ISO 639-3 to 639-1 mapping from TSV file."""
    iso_map = {}
    with open(mapping_tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) >= 2:
                iso_map[row[0]] = row[1]  # iso-3 -> iso-1
    return iso_map

def load_required_languages(req_lang_path):
    """Load required languages from text file with format [iso3] Language Name."""
    required_langs = {}
    with open(req_lang_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Match pattern [xxx] Language Name
            match = re.match(r'\[(\w+)\]\s+(.+)', line.strip())
            if match:
                iso3_code, lang_name = match.groups()
                required_langs[iso3_code] = lang_name
    return required_langs

def check_whisper_support(required_langs, iso_mapping, whisper_langs):
    """Check which required languages are supported by Whisper."""
    supported = []
    unsupported = []
    
    for iso3_code, lang_name in required_langs.items():
        # Get ISO 639-1 code if available
        iso1_code = iso_mapping.get(iso3_code, '')
        
        if iso1_code and iso1_code in whisper_langs:
            supported.append((iso3_code, iso1_code, lang_name))
        else:
            unsupported.append((iso3_code, iso1_code, lang_name))
            
    return supported, unsupported

def write_json_results(supported, unsupported, json_path):
    """Write results to a JSON file."""
    result_dict = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "supported": [
            {
                "name": name,
                "iso_639_3": iso3,
                "iso_639_1": iso1
            }
            for iso3, iso1, name in sorted(supported, key=lambda x: x[2])
        ],
        "unsupported": [
            {
                "name": name,
                "iso_639_3": iso3,
                "iso_639_1": iso1 if iso1 else None
            }
            for iso3, iso1, name in sorted(unsupported, key=lambda x: x[2])
        ],
        "summary": {
            "total": len(supported) + len(unsupported),
            "supported_count": len(supported),
            "unsupported_count": len(unsupported)
        }
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)

def write_text_results(supported, unsupported, output_path):
    """Write results to a text file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        # Write timestamp
        f.write(f"Language Support Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Write supported languages
        f.write("Supported Languages:\n")
        f.write("-------------------\n")
        for iso3, iso1, name in sorted(supported, key=lambda x: x[2]):
            f.write(f"{name:<30} (ISO-3: {iso3}, ISO-1: {iso1})\n")
        
        # Write unsupported languages
        f.write("\nUnsupported Languages:\n")
        f.write("---------------------\n")
        for iso3, iso1, name in sorted(unsupported, key=lambda x: x[2]):
            f.write(f"{name:<30} (ISO-3: {iso3}, ISO-1: {iso1 or 'N/A'})\n")
        
        # Write summary
        total = len(supported) + len(unsupported)
        supported_count = len(supported)
        f.write(f"\nSummary:\n")
        f.write(f"Total languages: {total}\n")
        f.write(f"Supported: {supported_count} ({supported_count/total*100:.1f}%)\n")
        f.write(f"Unsupported: {len(unsupported)} ({len(unsupported)/total*100:.1f}%)\n")

def main(mapping_tsv_path, req_lang_path, whisper_langs, output_path, json_path):
    """Main function to process language mappings."""
    # Load data
    iso_mapping = load_iso_mapping(mapping_tsv_path)
    required_langs = load_required_languages(req_lang_path)
    
    # Check support
    supported, unsupported = check_whisper_support(
        required_langs, iso_mapping, whisper_langs
    )
    
    # Write results to files
    write_text_results(supported, unsupported, output_path)
    write_json_results(supported, unsupported, json_path)
    
    print(f"Results have been written to {output_path}")
    # Print quick summary to console
    print(f"\nQuick Summary:")
    print(f"Total languages: {len(required_langs)}")
    print(f"Supported: {len(supported)}")
    print(f"Unsupported: {len(unsupported)}")

if __name__ == "__main__":
    # Example usage:
    mapping_tsv_path = "iso_mapping.tsv"  # Your TSV with ISO 639-3 to 639-1 mapping
    req_lang_path = "required_languages.txt"  # Your text file with required languages
    output_path = "language_support_analysis.txt"  # Text output file path
    json_path = "language_support_analysis.json"  # JSON output file path
    
    # Example Whisper supported languages dict (replace with actual)
    whisper_langs = {
        "en": "english",
        "zh": "chinese",
        "de": "german",
        "es": "spanish",
        "ru": "russian",
        "ko": "korean",
        "fr": "french",
        "ja": "japanese",
        "pt": "portuguese",
        "tr": "turkish",
        "pl": "polish",
        "ca": "catalan",
        "nl": "dutch",
        "ar": "arabic",
        "sv": "swedish",
        "it": "italian",
        "id": "indonesian",
        "hi": "hindi",
        "fi": "finnish",
        "vi": "vietnamese",
        "he": "hebrew",
        "uk": "ukrainian",
        "el": "greek",
        "ms": "malay",
        "cs": "czech",
        "ro": "romanian",
        "da": "danish",
        "hu": "hungarian",
        "ta": "tamil",
        "no": "norwegian",
        "th": "thai",
        "ur": "urdu",
        "hr": "croatian",
        "bg": "bulgarian",
        "lt": "lithuanian",
        "la": "latin",
        "mi": "maori",
        "ml": "malayalam",
        "cy": "welsh",
        "sk": "slovak",
        "te": "telugu",
        "fa": "persian",
        "lv": "latvian",
        "bn": "bengali",
        "sr": "serbian",
        "az": "azerbaijani",
        "sl": "slovenian",
        "kn": "kannada",
        "et": "estonian",
        "mk": "macedonian",
        "br": "breton",
        "eu": "basque",
        "is": "icelandic",
        "hy": "armenian",
        "ne": "nepali",
        "mn": "mongolian",
        "bs": "bosnian",
        "kk": "kazakh",
        "sq": "albanian",
        "sw": "swahili",
        "gl": "galician",
        "mr": "marathi",
        "pa": "punjabi",
        "si": "sinhala",
        "km": "khmer",
        "sn": "shona",
        "yo": "yoruba",
        "so": "somali",
        "af": "afrikaans",
        "oc": "occitan",
        "ka": "georgian",
        "be": "belarusian",
        "tg": "tajik",
        "sd": "sindhi",
        "gu": "gujarati",
        "am": "amharic",
        "yi": "yiddish",
        "lo": "lao",
        "uz": "uzbek",
        "fo": "faroese",
        "ht": "haitian creole",
        "ps": "pashto",
        "tk": "turkmen",
        "nn": "nynorsk",
        "mt": "maltese",
        "sa": "sanskrit",
        "lb": "luxembourgish",
        "my": "myanmar",
        "bo": "tibetan",
        "tl": "tagalog",
        "mg": "malagasy",
        "as": "assamese",
        "tt": "tatar",
        "haw": "hawaiian",
        "ln": "lingala",
        "ha": "hausa",
        "ba": "bashkir",
        "jw": "javanese",
        "su": "sundanese",
        "yue": "cantonese"
    }
    
    main(mapping_tsv_path, req_lang_path, whisper_langs, output_path, json_path)

