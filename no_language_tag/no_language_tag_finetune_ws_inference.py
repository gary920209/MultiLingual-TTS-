import copy
import sys

import nlp2
from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperModel, WhisperConfig, Seq2SeqTrainingArguments
from transformers import WhisperProcessor
from datasets import load_from_disk
from transformers.activations import ACT2FN
from module.args import parse_args
from module.data_processing import DataCollatorSpeechSeq2SeqWithPadding
from module.metric import cer_cal, wer_cal

from datetime import datetime


from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import gc
import torch
import torchaudio
from typing import Callable, Iterator, List, Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput, Seq2SeqModelOutput, BaseModelOutputWithPastAndCrossAttentions
from transformers.generation.configuration_utils import GenerationConfig
from transformers.models.whisper.modeling_whisper import shift_tokens_right, WhisperDecoder, WhisperEncoder, WhisperPositionalEmbedding, WhisperDecoderLayer
from torch.nn import CrossEntropyLoss
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa

# Peft
from transformers import Seq2SeqTrainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl, set_seed
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

# from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model, PeftConfig
from peft import prepare_model_for_int8_training

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

class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control

def load_peft_model_from_hub(peft_model_id):
    peft_config = PeftConfig.from_pretrained(peft_model_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        peft_config.base_model_name_or_path
    )
    model = PeftModel.from_pretrained(model, peft_model_id,  is_trainable=True) # the is_trainable parameter=true to make sure the model is tranable is we load the checkpoint instead of the base model. 
    
    print("Load model from hub successfully.")
    return model

class Whisper_Modified(WhisperForConditionalGeneration):
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
        lang_id = list(generation_config.lang_to_id.values())
        non_lang_mask[lang_id] = False
        logits[:, non_lang_mask] = -np.inf
        
        return logits.softmax(-1)

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

# for LrRescheduleTrainer
from functools import partial
from torch.optim.lr_scheduler import LambdaLR

class LrRescheduleTrainer(Seq2SeqTrainer):
    def __init__(self, specified_epoch, total_epoch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Add custom attributes here
        self.total_epoch = total_epoch
        self.specified_epoch = 0
        
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        
        self.lr_scheduler = self.get_linear_schedule_with_warmup(
            optimizer=self.optimizer if optimizer is None else optimizer,
            num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
        )
        return self.lr_scheduler

    def get_linear_schedule_with_warmup(self, optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
        """
        Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
        a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

        Args:
            optimizer ([`~torch.optim.Optimizer`]):
                The optimizer for which to schedule the learning rate.
            num_warmup_steps (`int`):
                The number of steps for the warmup phase.
            num_training_steps (`int`):
                The total number of training steps.
            last_epoch (`int`, *optional*, defaults to -1):
               The index of the last epoch when resuming training. 

        Return:
            `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
        """

        lr_lambda = partial(
            self._get_linear_schedule_with_warmup_lr_lambda,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        return LambdaLR(optimizer, lr_lambda, last_epoch)

    def _get_linear_schedule_with_warmup_lr_lambda(self, current_step: int, *, num_warmup_steps: int, num_training_steps: int):
        # The only difference
        current_step += num_training_steps * self.specified_epoch
        num_training_steps *= self.total_epoch

        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

def experiment(input_arg, model, processor, data_collator, data_train, data_test, time, output_dir, eval_only, corpus_wise):
    if not eval_only:
        training_args = Seq2SeqTrainingArguments(
            do_eval=False,
            output_dir=input_arg.get("output_dir", "."),
            length_column_name="lengths",
            group_by_length=input_arg["group_by_length"],
            per_device_train_batch_size=int(input_arg["batch"]),
            per_device_eval_batch_size=int(input_arg["batch"]),
            gradient_accumulation_steps=int(input_arg["grad_accum"]),
            eval_accumulation_steps=int(input_arg["grad_accum"]),
            evaluation_strategy="no",
            save_strategy="no",
            ddp_find_unused_parameters=True,
            resume_from_checkpoint=input_arg.get("checkpoint", False),
            overwrite_output_dir=input_arg.get("overwrite_output_dir", False),
            greater_is_better=False,
            metric_for_best_model="cer",
            num_train_epochs=input_arg.get("epoch", 5),
            fp16=True,
            logging_steps=input_arg.get("logging_steps", 10),
            learning_rate=input_arg.get("learning_rate", 4.7e-5),
            warmup_steps=input_arg.get("warmup_steps", 100),
            save_total_limit=input_arg.get("save_total_limit", 3),
            push_to_hub=False,
            report_to="none",
            weight_decay=input_arg.get("weight_decay", 0.02),
            remove_unused_columns=False,
            label_names=["labels"],
        )

        training_args.generation_max_length = 225

        trainer = LrRescheduleTrainer(
            specified_epoch=0,
            total_epoch=input_arg['epoch'],
            model=model,
            data_collator=data_collator,
            args=training_args,
            train_dataset=data_train,
            # eval_dataset=data_test,
            tokenizer=processor.feature_extractor,
            callbacks=[SavePeftModelCallback],
        )
        model.config.use_cache = False  

        trainer.train()
    ###################
    #     Evaluate    #
    ###################
    eval_dataloader = DataLoader(data_test, batch_size=int(input_arg["batch"]), collate_fn=data_collator)

    model.eval()
    model = model.to("cuda")
    label_list = []
    pred_list = []
    pred_results = []
    lang_id = list(model.generation_config.lang_to_id.values())
    weight_vis=torch.zeros((len(lang_id))).to("cuda")
    weight = None
    if corpus_wise:
        weight = torch.zeros((51865)).to("cuda")
        for batch in eval_dataloader:
            lang_distribution = model.detect_language_custom(input_features=batch["input_features"].to("cuda")).squeeze()
            weight += lang_distribution
        weight /= len(eval_dataloader)
        weight_vis = weight[lang_id]

    for step, batch in enumerate(tqdm(eval_dataloader)):
        with torch.no_grad():
            if weight != None:
                lang_distribution = weight
            else:
                lang_distribution = model.detect_language_custom(input_features=batch["input_features"].to("cuda")).squeeze()
                weight_vis += lang_distribution[lang_id]
            generated_tokens = (
                model.generate(
                    input_features=batch["input_features"].to("cuda"),
                    max_new_tokens=255,
                    decoder_input_ids=batch["labels"][:, :3].to("cuda"),
                    lang_distribution=lang_distribution,
                    task="transcribe"
                )
                .cpu()
                .numpy()
            )

            labels = batch["labels"].cpu().numpy()
            labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
            generated_tokens = torch.from_numpy(generated_tokens)
            pred_str = processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            label_str = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)

            pred_result = [[l, p, cer_cal([l], [p])] for l, p in zip(label_str, pred_str)]
            pred_results += pred_result

            pred_list += pred_str
            label_list += label_str
            pred_str = (" ").join(pred_str)
        del generated_tokens, labels, batch
        gc.collect()
    if not corpus_wise:
        weight_vis /= len(eval_dataloader)
    import matplotlib.pyplot as plt
    keys = list(model.generation_config.lang_to_id.keys())
    keys = [i.replace("<|", "").replace("|>", "") for i in keys]
    values = weight_vis.squeeze().tolist()
    plt.bar(range(len(keys)), values)
    plt.margins(x=0)
    plt.xlabel('Language')
    plt.ylabel('Prob')
    plt.title('Weight Distribution')
    plt.xticks(range(len(keys)), keys, rotation=90, fontsize=5)
    plt.savefig(f'{output_dir}/weight.png')
    nlp2.write_csv(pred_results, f'{output_dir}/pred.csv')
    cer = cer_cal(label_list, pred_list)
    wer = wer_cal(label_list, pred_list)
    print("********* Evaluation Result *********")
    print(f"cer: {cer}, wer: {wer}")
    print("*************************************")
    return model


def main(arg=None):
    set_seed(42)
    input_arg, other_arg = parse_args(sys.argv[1:]) if arg is None else parse_args(arg)
    ############
    #  Config  #
    ############
    size = input_arg["size"]
    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    input_arg["tokenize_config"] = f"openai/whisper-{size}"
    input_arg["model_config"] = f"openai/whisper-{size}"
    input_arg["group_by_length"] = True
    input_arg["cache_dir"] = "~/.cache"
    dropout = input_arg.get("dropout", 0.0)

    only_eval = input_arg.get("only_eval", False)
    corpus_wise = input_arg.get("corpus_wise", False)
    ############
    #  Model   #
    ############

    processor = WhisperProcessor.from_pretrained(
        input_arg["model_config"], task="transcribe", dropout=dropout, language=None
    )
    audio_feature_key = "input_ids"

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, audio_feature_key=audio_feature_key)

    # load from base model
    model = Whisper_Modified.from_pretrained(input_arg["model_config"])
    config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
    model = get_peft_model(model, config)
    model = model.to("cuda")
    
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    
    model.print_trainable_parameters()

    ############
    #  Dataset #
    ############
    weight=None
    dataset = load_dataset(
        "csv",
        data_files=input_arg["custom_set_train"],
        cache_dir=input_arg["cache_dir"],
    )
    dataset = dataset.filter(lambda e: nlp2.is_file_exist(e["path"]))
    data_train = dataset["train"]
    data_train = data_train.map(
        prepare_dataset_whisper,
        num_proc=1,
        fn_kwargs={"feature_extractor": processor.feature_extractor, "audio_feature_key": audio_feature_key},
    )
    data_train = data_train.map(encode_dataset, fn_kwargs={"processor": processor})

    dataset_test = load_dataset(
        "csv",
        data_files=input_arg["custom_set_test"],
        cache_dir=input_arg["cache_dir"],
        # cache_dir=None,
    )
    dataset_test = dataset_test.filter(lambda e: nlp2.is_file_exist(e["path"]))
    data_test = dataset_test["train"]

    data_test = data_test.map(
        prepare_dataset_whisper,
        num_proc=1,
        fn_kwargs={"feature_extractor": processor.feature_extractor, "audio_feature_key": audio_feature_key},
    )
    data_test = data_test.map(encode_dataset, fn_kwargs={"processor": processor})

    model = experiment(
        input_arg,
        model,
        processor,
        data_collator,
        data_train,
        data_test,
        time,
        output_dir=input_arg["output_dir"],
        eval_only=only_eval,
        corpus_wise=corpus_wise,
    )

if __name__ == "__main__":
    main()
