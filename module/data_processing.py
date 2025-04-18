from dataclasses import dataclass
from typing import Dict, List, Union, Any

import torch
import torchaudio
from transformers import Wav2Vec2Processor
from transformers.models.whisper.modeling_whisper import shift_tokens_right
from datasets import load_dataset, Audio
import numpy as np

def encode_dataset(batch, processor, phonemize=False, backend=None, separator=None):
    if not isinstance(batch["labels"], list):
        if phonemize:
            with processor.as_target_processor():
                if phonemize == "g2p":
                    batch["labels"] = processor(backend.encode(batch["labels"])).input_ids
                else:
                    batch["labels"] = processor(backend.phonemize([batch["labels"]], separator=separator)[0]).input_ids
        else:
            try:
                with processor.as_target_processor():
                    line = bytes(batch["labels"], "utf-8").decode("utf-8", "ignore")
                    batch["labels"] = processor(line).input_ids
            except Exception as e:
                line = bytes(batch["labels"], "utf-8").decode("utf-8", "ignore")
                batch["labels"] = processor.tokenizer(line).input_ids
    return batch


def prepare_dataset_hf(batch, processor, audio_feature_key):
    audio = batch["audio"]
    batch[audio_feature_key] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).get(audio_feature_key)[
        0
    ]
    batch["lengths"] = len(batch[audio_feature_key])
    if "sentence" in batch:
        batch["labels"] = batch["sentence"]
    else:
        batch["labels"] = batch["text"]
    return batch


def prepare_dataset_custom(batch, audio_feature_key):
    path = batch["path"]
    speech, sampling_rate = torchaudio.load(path)
    if sampling_rate != "16_000" or sampling_rate != "16000":
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16_000)
        batch[audio_feature_key] = resampler.forward(speech.squeeze(0)).numpy()
    else:
        batch[audio_feature_key] = speech.squeeze(0).numpy()
    batch["lengths"] = len(batch[audio_feature_key])
    if "sentence" in batch:
        batch["labels"] = batch["sentence"]
    else:
        batch["labels"] = batch["text"]
    return batch


def prepare_dataset_whisper(batch, feature_extractor, audio_feature_key):
    path = batch["path"]
    speech, sampling_rate = torchaudio.load(path)
    if sampling_rate != "16_000" or sampling_rate != "16000":
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16_000)
        batch[audio_feature_key] = resampler.forward(speech.squeeze(0)).numpy()
    else:
        batch[audio_feature_key] = speech.squeeze(0).numpy()
    # compute log-Mel input features from input audio array
    batch[audio_feature_key] = feature_extractor(batch[audio_feature_key], sampling_rate=16000).input_features[0]
    batch["lengths"] = len(batch[audio_feature_key])
    # # encode target text to label ids
    # batch["labels"] = tokenizer(batch["sentence"]).input_ids
    if "sentence" in batch:
        batch["labels"] = batch["sentence"]
    else:
        batch["labels"] = batch["text"]
    return batch


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    audio_feature_key: str = "input_values"
    # audio_feature_key: str = "input_ids"

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different length and need
        # different padding methods
     
        input_values = [{self.audio_feature_key: feature[self.audio_feature_key]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_values,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        return batch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    audio_feature_key: str = "input_features" 
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

        if "raw_audio" in features[0]:
            # Get raw audio and lid from features
            raw_audio = [feature["raw_audio"] for feature in features]
            lid = [feature["lid"] for feature in features]
        
        # Handle input features
        input_features = [{"input_features": feature[self.audio_feature_key]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Handle labels
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Cut bos token if present
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        # Add raw_audio, lid, and labels to batch
        if "raw_audio" in features[0]:
            batch["raw_audio"] = torch.tensor(np.array(raw_audio)).squeeze()
            batch["lid"] = lid
        batch["labels"] = labels
        
        return batch

@dataclass
class DataCollatorWeightedSum:
    model: Any
    processor: Any
    audio_feature_key: str = "input_features"
    weight: Any = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
        input_features = [{"input_features": feature[self.audio_feature_key]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # batch["labels"] = torch.tensor([feature["labels"] for feature in features])
        # batch["decoder_inputs_embeds"] = torch.tensor([feature["decoder_inputs_embeds"] for feature in features]).squeeze(0)
        input_features = [{"input_features": feature[self.audio_feature_key]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        decoder_input_ids = shift_tokens_right(
            batch["labels"], self.model.config.pad_token_id, self.model.config.decoder_start_token_id
        )
        all = isinstance(self.weight, dict)
        with torch.no_grad():
            embedding=self.model.get_decoder().get_input_embeddings()
            if self.weight == None:
                print(batch["input_features"])
                lang_distribution = self.model.detect_language_custom(batch["input_features"].to("cuda"), all=all)
            else:
                lang_distribution = self.weight[decoder_input_ids[0][2].item()] if all else self.weight
            token_embeddings = embedding(lang_distribution[0].nonzero()).squeeze(1)
            lang_distribution = lang_distribution[lang_distribution.nonzero(as_tuple=True)].view(lang_distribution.shape[0], -1)
            summation = embedding(decoder_input_ids.to("cuda"))
            summation[:,1,:] = torch.matmul(lang_distribution, token_embeddings)
            batch["weight"]=summation

        return batch

