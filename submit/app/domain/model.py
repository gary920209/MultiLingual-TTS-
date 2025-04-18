# import numpy as np
# from typing import List
# from app.domain.schemas.model import ModelSingleInput, ModelSingleOutput
# import torch


# class ModelController:
#     def __init__(self, model_id: str = "path_to_my_checkpoint", device: str = "cuda" if torch.cuda.is_available() else "cpu") -> None:
#         self.initialized = True
#         self.device = device
#         self.pretrained_model = # instaniate your model and load its weights here
#         self.sample_rate
        
#     def single_inference(self, input_data: ModelSingleInput) -> ModelSingleOutput:
#         """Run inference on a single audio sample.
        
#         Args:
#             input_data: ModelSingleInput containing audio, sample rate, and language if available
            
#         Returns:
#             ModelSingleOutput with transcription and confidence score
#         """
#         # Ensure audio is float32
#         audio = torch.from_numpy(np.array(input_data.audio).astype(np.float32))
#         audio = audio.to(self.device)

#         if self.sample_rate != input_data.sample_rate:
#             # if your model is not using 16kHz sampling rate
#             # you should resample the input audio

#         if input_data.language is not None and len(input_data.language) > 0:
           
#            # for this utterance, you have access to the true language identity.
#            # you can use it as a condition for your model or ignore it
#            pred_lid = input_data.language
#            pred_asr = 
#         else:
            
#             # for this utterance, you dont have access to the true language identity.
#             # you need to predict it and return it along with the predicted ASR text

#             pred_lid = 
#             pred_asr = 

#         return ModelSingleOutput(
#             language=pred_lid,
#             text=pred_asr,
#         )
import numpy as np
from typing import List, Callable, Iterator, Optional, Tuple, Union
from app.domain.schemas.model import ModelSingleInput, ModelSingleOutput
import torch
import torchaudio
from transformers import (
    WhisperProcessor, 
    Wav2Vec2ForSequenceClassification, 
    AutoFeatureExtractor, 
    WhisperForConditionalGeneration, 
    WhisperModel, 
    WhisperConfig, 
    GenerationConfig
)
from peft import PeftModel, PeftConfig
import os
import copy
import unicodedata
import re
import torch.nn  as nn
from transformers.modeling_outputs import (
    BaseModelOutput, 
    Seq2SeqLMOutput, 
    Seq2SeqModelOutput, 
    BaseModelOutputWithPastAndCrossAttentions
)
from transformers.models.whisper.modeling_whisper import (
    shift_tokens_right, 
    WhisperDecoder, 
    WhisperEncoder, 
    WhisperPositionalEmbedding, 
    WhisperDecoderLayer
)
from transformers.generation.configuration_utils import GenerationConfig
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask, 
    _prepare_4d_causal_attention_mask_for_sdpa
)

NEW_TOKEN_TO_ID = {
    "afr": 50327,
    "amh": 50334,
    "ara": 50272,
    "hye": 50312,
    "asm": 50350,
    "aze": 50304,
    "bak": 50355,
    "eus": 50310,
    "bel": 50330,
    "ben": 50302,
    "bos": 50315,
    "bre": 50309,
    "bul": 50292,
    "mya": 50346,
    "cat": 50270,
    "hrv": 50291,
    "ces": 50283,
    "dan": 50285,
    "nld": 50271,
    "eng": 50259,
    "est": 50307,
    "fin": 50277,
    "fra": 50265,
    "glg": 50319,
    "kat": 50329,
    "deu": 50261,
    "guj": 50333,
    "hat": 50339,
    "hau": 50354,
    "heb": 50279,
    "hin": 50276,
    "hun": 50286,
    "isl": 50311,
    "ind": 50275,
    "ita": 50274,
    "jpn": 50266,
    "kan": 50306,
    "kaz": 50316,
    "khm": 50323,
    "kor": 50264,
    "lao": 50336,
    "lav": 50301,
    "lin": 50353,
    "lit": 50293,
    "ltz": 50345,
    "mkd": 50308,
    "msa": 50282,
    "mal": 50296,
    "mlt": 50343,
    "mri": 50295,
    "mar": 50320,
    "ell": 50281,
    "mon": 50314,
    "nep": 50313,
    "oci": 50328,
    "pan": 50321,
    "fas": 50300,
    "pol": 50269,
    "por": 50267,
    "pus": 50340,
    "ron": 50284,
    "rus": 50263,
    "srp": 50303,
    "sna": 50324,
    "snd": 50332,
    "sin": 50322,
    "slk": 50298,
    "slv": 50305,
    "som": 50326,
    "spa": 50262,
    "sun": 50357,
    "swa": 50318,
    "swe": 50273,
    "fil": 50348,
    "tgk": 50331,
    "tam": 50287,
    "tat": 50351,
    "tel": 50299,
    "tha": 50289,
    "tur": 50268,
    "ukr": 50280,
    "urd": 50290,
    "uzb": 50337,
    "vie": 50278,
    "cym": 50297,
    "yor": 50325,
    "jav": 50356,
    "cmn": 50260,
    "abk": 51865,
    "ast": 51866,
    "bas": 51867,
    "ceb": 51868,
    "ckb": 51869,
    "chv": 51870,
    "div": 51871,
    "mhr": 51872,
    "myv": 51873,
    "epo": 51874,
    "ful": 51875,
    "lug": 51876,
    "grn": 51877,
    "cnh": 51878,
    "azz": 51879,
    "tos": 51880,
    "ibo": 51881,
    "ina": 51882,
    "gle": 51883,
    "kea": 51884,
    "kab": 51885,
    "kam": 51886,
    "kin": 51887,
    "kir": 51888,
    "lga": 51889,
    "luo": 51890,
    "nan": 51891,
    "frr": 51892,
    "kmr": 51893,
    "nya": 51894,
    "ori": 51895,
    "orm": 51896,
    "nso": 51897,
    "skr": 51898,
    "nbl": 51899,
    "sot": 51900,
    "ssw": 51901,
    "tok": 51902,
    "tso": 51903,
    "tsn": 51904,
    "uig": 51905,
    "umb": 51906,
    "hsb": 51907,
    "ven": 51908,
    "mrj": 51909,
    "wol": 51910,
    "xho": 51911,
    "sah": 51912,
    "xty": 51913,
    "yue": 51914,
    "zul": 51915
}

def prepare_dataset_whisper(batch, base_dir, feature_extractor, audio_feature_key):
    path = os.path.join(base_dir, batch["path"])
    speech, sampling_rate = torchaudio.load(path)
    batch["raw_audio"] = speech.squeeze(0).numpy()
    if sampling_rate != "16_000" and sampling_rate != "16000" and sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16_000)
        batch[audio_feature_key] = resampler.forward(speech.squeeze(0)).numpy()
    else:
        batch[audio_feature_key] = speech.squeeze(0).numpy()

    # compute log-Mel input features from input audio array
    batch[audio_feature_key] = feature_extractor(batch[audio_feature_key], sampling_rate=16000).input_features[0]
    batch["lengths"] = len(batch[audio_feature_key])

    # batch["labels"] = tokenizer(batch["sentence"]).input_ids
    if "sentence" in batch:
        batch["labels"] = batch["sentence"]
    else:
        batch["labels"] = batch["text"]
        
    batch["lid"] = path.split("/")[-3]
    return batch

def get_weight(processor, model, data_train):
    weight = torch.zeros_like(model.detect_language_custom(torch.Tensor(data_train[0]["input_ids"]).unsqueeze(0).to("cuda")))
    for batch in data_train:
        lang_distribution = model.detect_language_custom(torch.Tensor(batch["input_ids"]).unsqueeze(0).to("cuda"))
        weight += lang_distribution
    weight /= len(data_train)
    return weight

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
    if len(batch["labels"]) > 448:
        batch["labels"] = batch["labels"][:448]
    return batch


def load_peft_model_from_hub(peft_model_id):
    peft_config = PeftConfig.from_pretrained(peft_model_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        peft_config.base_model_name_or_path
    )
    model = PeftModel.from_pretrained(model, peft_model_id,  is_trainable=True) # the is_trainable parameter=true to make sure the model is tranable is we load the checkpoint instead of the base model. 
    
    print("Load model from hub successfully.")
    return model


def load_model_state(output_dir, size):
    """
    Load model state using base Whisper and saved components.
    """
    # 1. Load processor/tokenizer from base Whisper
    processor = WhisperProcessor.from_pretrained(output_dir, task="transcribe")
    
    # 2. Load embeddings if they exist
    embeddings_path = os.path.join(output_dir, 'language_embeddings.pt')
    embeddings = None
    # if os.path.exists(embeddings_path):
    #     embeddings = torch.load(embeddings_path)
    
    # 3. Initialize model from base Whisper
    try:
        model = Whisper_Modified.from_pretrained(
            f"openai/whisper-{size}",
            new_embedding=embeddings['weight'] if embeddings else None,
            language_tokens=embeddings['tokens_embed'] if embeddings else None,
            ignore_mismatched_sizes=True  # Add this if there are size mismatches
        )
    except TypeError as e:
        # If the above fails, try loading with minimal parameters
        model = Whisper_Modified.from_pretrained(
            f"openai/whisper-{size}",
            new_embedding=embeddings['weight'] if embeddings else None,
            language_tokens=embeddings['tokens_embed'] if embeddings else None
        )
    
    model.resize_token_embeddings(len(processor.tokenizer))

    # 4. Load saved proj_out weights
    proj_out_path = os.path.join(output_dir, "proj_out.pt")
    if os.path.exists(proj_out_path):
        proj_out_state = torch.load(proj_out_path)
        model.proj_out.load_state_dict(proj_out_state)
    
    # 5. Load and apply LoRA weights
    model = PeftModel.from_pretrained(model, os.path.join(output_dir, "adapter_model"))
    
    return model, processor


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

class Whisper_Modified(WhisperForConditionalGeneration):
    def __init__(self, config: WhisperConfig, new_embedding: torch.Tensor=None, language_tokens: torch.Tensor=None):
        super().__init__(config)
        self.model = WhisperModel(config)
        self.proj_out = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if new_embedding == None:
            self.weight = None
        elif isinstance(new_embedding, dict):
            new_lang_tokens = {k: v for k, v in new_embedding.items() if int(k) >= 51865}
            init_tensor = torch.zeros((len(new_lang_tokens), len(new_lang_tokens[list(new_lang_tokens.keys())[0]])))
            for key, value in new_lang_tokens.items():
                init_tensor[int(key) - 51865] = torch.tensor(value).unsqueeze(0)
            self.weight = nn.Parameter(init_tensor)
        else:
            self.weight = nn.Parameter(torch.tensor(new_embedding))
        self.tokens_embed = torch.tensor(language_tokens).to("cuda") if language_tokens is not None else None
        
        # Initialize weights and apply final processing
        self.post_init()
    
    def generate(
        self,
        input_features: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor= None,
        stopping_criteria= None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: bool = False,
        return_timestamps: Optional[bool] = None,
        task: Optional[str] = None,
        language: Optional[str] = None,
        is_multilingual: Optional[bool] = None,
        prompt_ids: Optional[torch.Tensor] = None,
        prompt_condition_type: Optional[str] = None,  # first-segment, all-segments
        condition_on_prev_tokens: Optional[bool] = None,
        temperature: Optional[Union[float, Tuple[float, ...]]] = None,
        compression_ratio_threshold: Optional[float] = None,
        logprob_threshold: Optional[float] = None,
        no_speech_threshold: Optional[float] = None,
        num_segment_frames: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        time_precision: float = 0.02,
        return_token_timestamps: Optional[bool] = None,
        return_segments: bool = False,
        return_dict_in_generate: Optional[bool] = None,
        lang_distribution=None,
        **kwargs,
    ):
        self.lang_distribution = lang_distribution
        # 0. deprecate old inputs
        if "inputs" in kwargs:
            input_features = kwargs.pop("inputs")
            warnings.warn(
                "The input name `inputs` is deprecated. Please make sure to use `input_features` instead.",
                FutureWarning,
            )
        # 1. copy generation config
        if generation_config is None:
            generation_config = copy.deepcopy(self.generation_config)
        else:
            generation_config = copy.deepcopy(generation_config)

        # 2. set global generate variables
        input_stride = self.model.encoder.conv1.stride[0] * self.model.encoder.conv2.stride[0]
        num_segment_frames = input_stride * self.config.max_source_positions
        batch_size, total_input_frames = self._retrieve_total_input_frames(
            input_features=input_features, input_stride=input_stride, kwargs=kwargs
        )
        is_shortform = total_input_frames <= num_segment_frames

        if is_shortform:
            # warn user of ignored inputs
            self._maybe_warn_unused_inputs(
                condition_on_prev_tokens=condition_on_prev_tokens,
                temperature=temperature,
                compression_ratio_threshold=compression_ratio_threshold,
                logprob_threshold=logprob_threshold,
                no_speech_threshold=no_speech_threshold,
                total_input_frames=total_input_frames,
            )
        # 3. Make sure generation config is correctly set
        # Make sure the generation config is correctly set depending on whether timestamps are to be returned or not
        self._set_return_outputs(
            return_dict_in_generate=return_dict_in_generate,
            return_token_timestamps=return_token_timestamps,
            is_shortform=is_shortform,
            logprob_threshold=logprob_threshold,
            generation_config=generation_config,
        )
        self._set_return_timestamps(
            return_timestamps=return_timestamps, is_shortform=is_shortform, generation_config=generation_config
        )
        self._set_language_and_task(
            language=language, task=task, is_multilingual=is_multilingual, generation_config=generation_config
        )
        self._set_token_ids(generation_config=generation_config, config=self.config, kwargs=kwargs)
        self._set_num_frames(
            return_token_timestamps=return_token_timestamps, generation_config=generation_config, kwargs=kwargs
        )
        self._set_thresholds_and_condition(
            generation_config=generation_config,
            logprob_threshold=logprob_threshold,
            compression_ratio_threshold=compression_ratio_threshold,
            no_speech_threshold=no_speech_threshold,
            condition_on_prev_tokens=condition_on_prev_tokens,
        )
        self._set_prompt_condition_type(
            generation_config=generation_config,
            prompt_condition_type=prompt_condition_type,
        )

        # pass self.config for backward compatibility
        init_tokens = self._retrieve_init_tokens(
            input_features,
            generation_config=generation_config,
            config=self.config,
            num_segment_frames=num_segment_frames,
            kwargs=kwargs,
        )
        # passing `decoder_input_ids` is deprecated - the only exception is for assisted generation
        # where the input ids are handled explicitly by the generate method
        self._check_decoder_input_ids(kwargs=kwargs)

        # 3. Retrieve logits processors
        begin_index = len(init_tokens)
        logits_processor = self._retrieve_logit_processors(
            generation_config=generation_config,
            logits_processor=logits_processor,
            begin_index=begin_index,  # begin index is index of first generated decoder token
            is_shortform=is_shortform,
            num_beams=kwargs.get("num_beams", 1),
        )

        # 5. If we're in shortform mode, simple generate the whole input at once and return the output
        if is_shortform:
            if temperature is not None:
                kwargs["temperature"] = temperature

            decoder_input_ids = kwargs.pop("decoder_input_ids", None)
            if decoder_input_ids is None:
                one_tensor = torch.ones((batch_size, 1), device=self.device, dtype=torch.long)
                decoder_input_ids = torch.cat([t * one_tensor for t in init_tokens], dim=-1)

            if prompt_ids is not None:
                decoder_input_ids = torch.cat(
                    [prompt_ids[None].repeat(decoder_input_ids.shape[0], 1), decoder_input_ids], dim=-1
                )

            if kwargs.get("max_new_tokens", 0) + decoder_input_ids.shape[-1] > self.config.max_target_positions:
                max_new_tokens = kwargs.get("max_new_tokens", 0)
                raise ValueError(
                    f"The length of `decoder_input_ids` equal `prompt_ids` plus special start tokens is {decoder_input_ids.shape[-1]}, and the `max_new_tokens` "
                    f"is {max_new_tokens}. Thus, the combined length of "
                    f"`decoder_input_ids` and `max_new_tokens` is: {max_new_tokens + decoder_input_ids.shape[-1]}. This exceeds the "
                    f"`max_target_positions` of the Whisper model: {self.config.max_target_positions}. "
                    "You should either reduce the length of your prompt, or reduce the value of `max_new_tokens`, "
                    f"so that their combined length is less than {self.config.max_target_positions}."
                )
            
            
            outputs = super().generate(
                input_features,
                generation_config=generation_config,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                synced_gpus=synced_gpus,
                decoder_input_ids=decoder_input_ids,
                # decoder_inputs_embeds=summation,
                **kwargs,
            )

            if generation_config.return_token_timestamps and hasattr(generation_config, "alignment_heads"):
                outputs["token_timestamps"] = self._extract_token_timestamps(
                    outputs, generation_config.alignment_heads, num_frames=generation_config.num_frames
                )

            return outputs

        # 6. Else we're in longform mode which is more complex.
        # We need to chunk the audio input depending on when the model generates timestamp tokens

        # 6.1 Set and retrieve global longform generation variables
        self._set_condition_on_prev_tokens(
            condition_on_prev_tokens=condition_on_prev_tokens, generation_config=generation_config
        )

        timestamp_begin = generation_config.no_timestamps_token_id + 1
        temperatures = [temperature] if not isinstance(temperature, (list, tuple)) else temperature
        temperature = temperatures[0]
        batch_size = input_features.shape[0]

        max_frames, seek = self._retrieve_max_frames_and_seek(
            batch_size=batch_size, attention_mask=attention_mask, total_input_frames=total_input_frames
        )

        # 6.2 Preppare running variables, list for generation
        cur_bsz = batch_size
        current_segments = self._prepare_segments(
            prompt_ids=prompt_ids,
            batch_size=batch_size,
            generation_config=generation_config,
        )

        batch_idx_map = list(range(batch_size))
        do_condition_on_prev_tokens = [condition_on_prev_tokens for _ in range(batch_size)]

        # 6.2 Transcribe audio until we reach the end of all input audios
        while (seek < max_frames).any():
            # 6.3 NOTE: When in longform transcription mode and batch size > 1 we need to dynamically reduce the batch size during the loop
            # in case one audio finished earlier than another one. Thus, we need to keep a table of "previous-index-2-current-index" in order
            # to know which original audio is being decoded
            # Set updated index map, duration of previously decoded chunks and number of max frames of current decoding chunk
            input_features, cur_bsz, batch_idx_map = self._maybe_reduce_batch(
                input_features=input_features,
                seek=seek,
                max_frames=max_frames,
                cur_bsz=cur_bsz,
                batch_idx_map=batch_idx_map,
            )
            time_offset = seek * time_precision / input_stride
            seek_num_frames = (max_frames - seek).clamp(max=num_segment_frames)

            # 6.4 cut out next 30s segment from input features
            segment_input = self._get_input_segment(
                input_features=input_features,
                seek=seek,
                seek_num_frames=seek_num_frames,
                num_segment_frames=num_segment_frames,
                cur_bsz=cur_bsz,
                batch_idx_map=batch_idx_map,
            )

            # 6.5 prepare decoder input ids
            suppress_tokens = _get_attr_from_logit_processors(
                logits_processor, SuppressTokensLogitsProcessor, "suppress_tokens"
            )
            decoder_input_ids, kwargs = self._prepare_decoder_input_ids(
                cur_bsz=cur_bsz,
                init_tokens=init_tokens,
                current_segments=current_segments,
                batch_idx_map=batch_idx_map,
                do_condition_on_prev_tokens=do_condition_on_prev_tokens,
                prompt_ids=prompt_ids,
                generation_config=generation_config,
                config=self.config,
                device=segment_input.device,
                suppress_tokens=suppress_tokens,
                kwargs=kwargs,
            )

            # 6.6 set max new tokens or max length
            kwargs = self._set_max_new_tokens_and_length(
                config=self.config,
                decoder_input_ids=decoder_input_ids,
                generation_config=generation_config,
                kwargs=kwargs,
            )

            # 6.7 Set current `begin_index` for all logit processors
            for proc in logits_processor:
                if hasattr(proc, "set_begin_index"):
                    proc.set_begin_index(decoder_input_ids.shape[-1])

            # 6.8 Run generate with fallback
            seek_sequences, seek_outputs, should_skip, do_condition_on_prev_tokens = self.generate_with_fallback(
                segment_input=segment_input,
                decoder_input_ids=decoder_input_ids,
                cur_bsz=cur_bsz,
                batch_idx_map=batch_idx_map,
                seek=seek,
                num_segment_frames=num_segment_frames,
                max_frames=max_frames,
                temperatures=temperatures,
                generation_config=generation_config,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                synced_gpus=synced_gpus,
                return_token_timestamps=return_token_timestamps,
                do_condition_on_prev_tokens=do_condition_on_prev_tokens,
                kwargs=kwargs,
            )

            # 6.9 In every generated sequence, split by timestamp tokens and extract segments
            for i, seek_sequence in enumerate(seek_sequences):
                prev_i = batch_idx_map[i]

                if should_skip[i]:
                    seek[prev_i] += seek_num_frames[prev_i]
                    continue

                segments, segment_offset = self._retrieve_segment(
                    seek_sequence=seek_sequence,
                    seek_outputs=seek_outputs,
                    time_offset=time_offset,
                    timestamp_begin=timestamp_begin,
                    seek_num_frames=seek_num_frames,
                    time_precision=time_precision,
                    input_stride=input_stride,
                    prev_idx=prev_i,
                    idx=i,
                    return_token_timestamps=return_token_timestamps,
                )

                current_segments[prev_i] += segments
                seek[prev_i] += segment_offset

        # 7. Once all segments are added to the list of all segments, called `current_segments`, we extract the predicted
        # output tokens from the list of dicts. If we use batch size > 1, we make sure to pad the output
        final_segments = (
            [x[1:] for x in current_segments]
            if (prompt_ids is not None and generation_config.prompt_condition_type == "first-segment")
            else current_segments
        )
        sequences = _pad_to_max_length(final_segments, generation_config.pad_token_id, padding="right")

        # 8. If we return all segments, the predicted output sequences are put under `"sequences"`.
        if return_segments:
            return {"sequences": sequences, "segments": final_segments}

        return sequences

    def detect_language_custom(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[Union[torch.FloatTensor, BaseModelOutput]] = None,
        generation_config: Optional[GenerationConfig] = None,
        num_segment_frames: int = 3000,
        top_k: int = None
    ) -> torch.Tensor:
        if input_features is None and encoder_outputs is None:
            raise ValueError("You have to specify either `input_features` or `encoder_outputs`")
        elif input_features is not None and encoder_outputs is not None:
            raise ValueError("Make sure to specificy only one of `input_features` or `encoder_outputs` - not both!")
        elif input_features is not None:
            inputs = {"input_features": input_features[:, :, :num_segment_frames]}
            batch_size = input_features.shape[0]
        elif encoder_outputs is not None:
            inputs = {"encoder_outputs": encoder_outputs}
            batch_size = (
                encoder_outputs[0].shape[0] if isinstance(encoder_outputs, BaseModelOutput) else encoder_outputs[0]
            )

        generation_config = generation_config or self.generation_config
        decoder_input_ids = (
            torch.ones((batch_size, 1), device=self.device, dtype=torch.long)
            * generation_config.decoder_start_token_id
        )

        with torch.no_grad():
            logits = self(**inputs, decoder_input_ids=decoder_input_ids).logits[:, -1]

        non_lang_mask = torch.ones_like(logits[0], dtype=torch.bool)
        non_lang_mask[list(generation_config.lang_to_id.values())] = False

        logits[:, non_lang_mask] = -np.inf

        return logits.softmax(-1)

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        use_cache=None,
        encoder_outputs=None,
        attention_mask=None,
        decoder_attention_mask=None,
        **kwargs,
    ):
        decoder_position_ids = None
        if decoder_attention_mask is not None:
            decoder_position_ids = (decoder_attention_mask.cumsum(-1) - 1).clamp(min=0)

        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]


            # Some generation methods already pass only the last input ID
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1


            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]


            if decoder_position_ids is not None and decoder_position_ids.shape[1] > decoder_input_ids.shape[1]:
                decoder_position_ids = decoder_position_ids[:, remove_prefix_length:]

        if past_key_values is None:
            embedding = self.get_decoder().get_input_embeddings()
            lang_distribution = self.lang_distribution
            if len(lang_distribution.shape) < 2:
                lang_distribution = lang_distribution.unsqueeze(0)
            token_embeddings = embedding(lang_distribution[0].nonzero()).squeeze(1)
            lang_distribution = lang_distribution[lang_distribution.nonzero(as_tuple=True)].view(lang_distribution.shape[0], -1)
            summation = embedding(decoder_input_ids)
            batch_size = summation.shape[0]
            zeros = torch.zeros((batch_size, 1, 1280)).to("cuda")
            summation = torch.cat([summation[:, :1, :], zeros, summation[:, 1:, :]], dim=1)
            summation[:,1,:] = torch.matmul(lang_distribution, token_embeddings)
            return {
                "encoder_outputs": encoder_outputs,
                "past_key_values": past_key_values,
                "decoder_inputs_embeds": summation,
                "use_cache": use_cache,
                "decoder_attention_mask": decoder_attention_mask,
                "decoder_position_ids": decoder_position_ids,
            }
        else:
            return {
                "encoder_outputs": encoder_outputs,
                "past_key_values": past_key_values,
                "decoder_input_ids": decoder_input_ids,
                "use_cache": use_cache,
                "decoder_attention_mask": decoder_attention_mask,
                "decoder_position_ids": decoder_position_ids,
            }

class ModelController:
    def __init__(self, model_id: str = "/Users/garylee/Desktop/project/MLSUPERB2-Challenge/submit/checkpoint-14710", device: str = "cuda" if torch.cuda.is_available() else "cpu") -> None:
        self.initialized = True
        self.device = device
        self.sample_rate = 16000 # Whisper uses 16kHz
        
        # Load processors and models
        self.processor = WhisperProcessor.from_pretrained(
            model_id,
            task="transcribe"
        )
        
        # Load main ASR model
        self.model = self.load_model_state(model_id)
        self.model.to(device)
        self.model.eval()
        
        # Load LID model
        lid_model_id = "facebook/mms-lid-4017"
        self.lid_processor = AutoFeatureExtractor.from_pretrained(lid_model_id)
        self.lid_model = Wav2Vec2ForSequenceClassification.from_pretrained(lid_model_id)
        self.lid_model.to(device)
        self.lid_model.eval()

    def load_model_state(self, checkpoint_dir):
        """Load the modified Whisper model with LoRA adapters"""
        # Load base Whisper model
        model = Whisper_Modified.from_pretrained(
            "openai/whisper-large-v3", 
            ignore_mismatched_sizes=True
        )
        
        # Load LoRA adapters
        model = PeftModel.from_pretrained(
            model,
            os.path.join(checkpoint_dir, "adapter_model")
        )
        
        return model
        
    def normalize_text(self, text: str, lang: str) -> str:
        """Normalize the output text based on language"""
        # Remove spaces for specific languages
        if lang in ['cmn', 'jpn', 'tha']:
            text = text.replace(" ", "")
        
        # Remove punctuation and convert to uppercase
        text = ''.join(c for c in text if not unicodedata.category(c).startswith('P'))
        return text.upper()

    def language_identification(self, audio_input, lang_list=None):
        """Perform language identification"""
        inputs = self.lid_processor(
            audio_input, 
            sampling_rate=self.sample_rate, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.lid_model(**inputs)
            
        predictions = outputs.logits.softmax(dim=-1)
        if lang_list:
            # Filter predictions to only include languages in lang_list
            lang_ids = [self.lid_model.config.label2id[lang] for lang in lang_list]
            mask = torch.zeros_like(predictions[0])
            mask[lang_ids] = 1
            predictions = predictions * mask
            predictions = predictions / predictions.sum()
            
        predicted_id = predictions.argmax().item()
        predicted_lang = self.lid_model.config.id2label[predicted_id]
        confidence = predictions[0][predicted_id].item()
        
        return predicted_lang, confidence

    def single_inference(self, input_data: ModelSingleInput) -> ModelSingleOutput:
        """Run inference on a single audio sample."""
        # Convert audio to correct format
        audio = torch.from_numpy(np.array(input_data.audio).astype(np.float32))
        audio = audio.to(self.device)

        # Resample if necessary
        if input_data.sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=input_data.sample_rate,
                new_freq=self.sample_rate
            )
            audio = resampler(audio)

        # Process audio features
        features = self.processor(
            audio.cpu().numpy(),  
            sampling_rate=self.sample_rate, 
            return_tensors="pt"
        ).input_features.to(self.device)

        if input_data.language is not None and len(input_data.language) > 0:
            # Use provided language
            pred_lid = input_data.language
            lang_token_id = NEW_TOKEN_TO_ID.get(pred_lid, NEW_TOKEN_TO_ID['eng'])
            lang_distribution = torch.zeros(len(NEW_TOKEN_TO_ID)).to(self.device)
            lang_distribution[lang_token_id] = 1.0
        else:
            # Predict language
            pred_lid, _ = self.language_identification(audio)
            lang_distribution = self.model.detect_language_custom(
                input_features=features
            ).squeeze()

        # Generate transcription
        with torch.no_grad():
            generated_tokens = self.model.generate(
                input_features=features,
                max_new_tokens=255,
                lang_distribution=lang_distribution,
                task="transcribe"
            )
            
        # Decode and normalize text
        pred_asr = self.processor.decode(generated_tokens[0], skip_special_tokens=True)
        pred_asr = self.normalize_text(pred_asr, pred_lid)

        return ModelSingleOutput(
            language=pred_lid,
            text=pred_asr
        )