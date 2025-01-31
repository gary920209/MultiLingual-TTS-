import json

# Hardcoded Whisper language mappings
WHISPER_LANG_TO_ID = {
    "en": 50259,  # English
    "zh": 50260,  # Chinese
    "de": 50261,  # German
    "es": 50262,  # Spanish
    "ru": 50263,  # Russian
    "ko": 50264,  # Korean
    "fr": 50265,  # French
    "ja": 50266,  # Japanese
    "pt": 50267,  # Portuguese
    "tr": 50268,  # Turkish
    "pl": 50269,  # Polish
    "ca": 50270,  # Catalan
    "nl": 50271,  # Dutch
    "ar": 50272,  # Arabic
    "sv": 50273,  # Swedish
    "it": 50274,  # Italian
    "id": 50275,  # Indonesian
    "hi": 50276,  # Hindi
    "fi": 50277,  # Finnish
    "vi": 50278,  # Vietnamese
    "he": 50279,  # Hebrew
    "uk": 50280,  # Ukrainian
    "el": 50281,  # Greek
    "ms": 50282,  # Malay
    "cs": 50283,  # Czech
    "ro": 50284,  # Romanian
    "da": 50285,  # Danish
    "hu": 50286,  # Hungarian
    "ta": 50287,  # Tamil
    "no": 50288,  # Norwegian
    "th": 50289,  # Thai
    "ur": 50290,  # Urdu
    "hr": 50291,  # Croatian
    "bg": 50292,  # Bulgarian
    "lt": 50293,  # Lithuanian
    "la": 50294,  # Latin
    "mi": 50295,  # Maori
    "ml": 50296,  # Malayalam
    "cy": 50297,  # Welsh
    "sk": 50298,  # Slovak
    "te": 50299,  # Telugu
    "fa": 50300,  # Persian
    "lv": 50301,  # Latvian
    "bn": 50302,  # Bengali
    "sr": 50303,  # Serbian
    "az": 50304,  # Azerbaijani
    "sl": 50305,  # Slovenian
    "kn": 50306,  # Kannada
    "et": 50307,  # Estonian
    "mk": 50308,  # Macedonian
    "br": 50309,  # Breton
    "eu": 50310,  # Basque
    "is": 50311,  # Icelandic
    "hy": 50312,  # Armenian
    "ne": 50313,  # Nepali
    "mn": 50314,  # Mongolian
    "bs": 50315,  # Bosnian
    "kk": 50316,  # Kazakh
    "sq": 50317,  # Albanian
    "sw": 50318,  # Swahili
    "gl": 50319,  # Galician
    "mr": 50320,  # Marathi
    "pa": 50321,  # Punjabi
    "si": 50322,  # Sinhala
    "km": 50323,  # Khmer
    "sn": 50324,  # Shona
    "yo": 50325,  # Yoruba
    "so": 50326,  # Somali
    "af": 50327,  # Afrikaans
    "oc": 50328,  # Occitan
    "ka": 50329,  # Georgian
    "be": 50330,  # Belarusian
    "tg": 50331,  # Tajik
    "sd": 50332,  # Sindhi
    "gu": 50333,  # Gujarati
    "am": 50334,  # Amharic
    "yi": 50335,  # Yiddish
    "lo": 50336,  # Lao
    "uz": 50337,  # Uzbek
    "fo": 50338,  # Faroese
    "ht": 50339,  # Haitian Creole
    "ps": 50340,  # Pashto
    "tk": 50341,  # Turkmen
    "nn": 50342,  # Nynorsk
    "mt": 50343,  # Maltese
    "sa": 50344,  # Sanskrit
    "lb": 50345,  # Luxembourgish
    "my": 50346,  # Myanmar
    "bo": 50347,  # Tibetan
    "tl": 50348,  # Tagalog
    "mg": 50349,  # Malagasy
    "as": 50350,  # Assamese
    "tt": 50351,  # Tatar
    "haw": 50352,  # Hawaiian
    "ln": 50353,  # Lingala
    "ha": 50354,  # Hausa
    "ba": 50355,  # Bashkir
    "jw": 50356,  # Javanese
    "su": 50357,  # Sundanese
}

def create_mapping(input_file, output_file):
    # Load the language list
    with open(input_file, 'r', encoding='utf-8') as f:
        language_list = json.load(f)
    
    # Initialize the mapping dictionary
    new_token_to_id = {}
    next_token_id = 51865  # Starting ID for unsupported languages
    
    # Process supported languages
    for lang in language_list["supported"]:
        iso_639_1 = lang["iso_639_1"]
        iso_639_3 = lang["iso_639_3"]
        
        if iso_639_1 in WHISPER_LANG_TO_ID:
            # Use existing Whisper token ID
            new_token_to_id[iso_639_3] = WHISPER_LANG_TO_ID[iso_639_1]
    
    # Process unsupported languages
    for lang in language_list["unsupported"]:
        iso_639_1 = lang["iso_639_1"]
        iso_639_3 = lang["iso_639_3"]
        
        if iso_639_1 not in WHISPER_LANG_TO_ID:
            # Assign new token ID
            new_token_to_id[iso_639_3] = next_token_id
            next_token_id += 1
    
    # Save the mapping
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(new_token_to_id, f, indent=4, ensure_ascii=False)
    
    # Print statistics
    existing_langs = sum(1 for id in new_token_to_id.values() if id < 51865)
    new_langs = sum(1 for id in new_token_to_id.values() if id >= 51865)
    print(f"\nMapping created successfully!")
    print(f"Total languages: {len(new_token_to_id)}")
    print(f"Existing Whisper languages: {existing_langs}")
    print(f"New languages added: {new_langs}")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate language mapping for Whisper fine-tuning')
    parser.add_argument('--input', type=str, required=True, help='Path to the languages JSON file')
    parser.add_argument('--output', type=str, required=True, help='Path to save the NEW_TOKEN_TO_ID mapping')
    args = parser.parse_args()
    
    create_mapping(args.input, args.output)
