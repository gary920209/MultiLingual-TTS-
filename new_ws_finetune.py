import copy
import sys, os

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
from torch import nn
from tqdm import tqdm
import numpy as np
import gc
import torch
import torchaudio
from typing import Callable, Iterator, List, Optional, Tuple, Union, Dict
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

from module.mapping import NEW_TOKEN_TO_ID

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

def get_weight(processor, model, data_train, all=False):
    weight = torch.zeros_like(model.detect_language_custom(torch.Tensor(data_train[0]["input_ids"]).unsqueeze(0).to("cuda"), all=all))
    if all:
        weights={}
        for key, value in NEW_TOKEN_TO_ID.items():
            weights[value] = [weight, 0]
        for batch in tqdm(data_train):
            lang_distribution = model.detect_language_custom(torch.Tensor(batch["input_ids"]).unsqueeze(0).to("cuda"), all=True)
            weights[batch["labels"][1]][0] += lang_distribution
            weights[batch["labels"][1]][1] += 1
        for key, value in NEW_TOKEN_TO_ID.items():
            weights[value] = weights[value][0] / weights[value][1]
            # weights[value] = weights[value][0] / 1
        return weights
    else:
        for batch in data_train:
            lang_distribution = model.detect_language_custom(torch.Tensor(batch["input_ids"]).unsqueeze(0).to("cuda"))
            weight += lang_distribution
        weight /= len(data_train)
        return weight

def encode_dataset(batch, processor, all=False, phonemize=False, backend=None, separator=None):
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
    if all:
        lang=batch["path"].split("/")[-3]
        lang_id=NEW_TOKEN_TO_ID[lang]
        batch["labels"] = batch["labels"][:1] + [lang_id] + batch["labels"][1:]
    else:
        batch["labels"] = batch["labels"][:1] + [51865] + batch["labels"][1:]
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

class Whisper_Modified(WhisperForConditionalGeneration):
    def __init__(self, config: WhisperConfig, new_embedding: torch.Tensor=None, language_tokens: torch.Tensor=None):
        super().__init__(config)
        self.model = WhisperModel(config)
        self.proj_out = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if new_embedding == None:
            self.weight = None
        elif isinstance(new_embedding, dict):
            init_tensor = torch.zeros((len(new_embedding.keys()), len(new_embedding[51865])))
            for key, value in new_embedding.items():
                init_tensor[int(key) - 51865] = torch.tensor(value).unsqueeze(0)
            self.weight = nn.Parameter(init_tensor)
        else:
            self.weight = nn.Parameter(torch.tensor(new_embedding))
        self.tokens_embed = torch.tensor(language_tokens).to("cuda") if language_tokens is not None else None

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        decoder_position_ids: Optional[Tuple[torch.LongTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
        
        if self.weight is not None and decoder_input_ids[0].shape[0] >= 4:
            embedding=self.get_decoder().get_input_embeddings()
            token_embeddings = embedding(self.tokens_embed)
            summation = embedding(decoder_input_ids)
            if len(self.weight.shape) > 1:
                if labels is not None:
                    summation[:,2,:] = torch.matmul(self.weight[decoder_input_ids[0][2] - 51865], token_embeddings)
                else:
                    summation[:,1,:] = torch.matmul(self.weight[decoder_input_ids[0][1] - 51865], token_embeddings)
            else:
                if labels is not None:
                    summation[:,2,:] = torch.matmul(self.weight, token_embeddings)
                else:
                    summation[:,1,:] = torch.matmul(self.weight, token_embeddings)
            decoder_inputs_embeds = summation
            decoder_input_ids = None

        outputs = self.model(
            input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_position_ids=decoder_position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.proj_out(outputs[0])

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.reshape(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def detect_language_custom(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[Union[torch.FloatTensor, BaseModelOutput]] = None,
        generation_config: Optional[GenerationConfig] = None,
        num_segment_frames: int = 3000,
        all: bool = False
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
        if all:
            lang_id.extend([i for i in range(51865, 51895)])
        else:
            lang_id.append(51865)
        non_lang_mask[lang_id] = False

        logits[:, non_lang_mask] = -np.inf
        
        return logits.softmax(-1)


def experiment(input_arg, model, processor, data_collator, data_train, data_test, time, output_dir, weight):
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
        dataloader_pin_memory=False
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

    for step, batch in enumerate(tqdm(eval_dataloader)):
        with torch.no_grad():
            generated_tokens = (
                model.generate(
                    input_features=batch["input_features"].to("cuda"),
                    max_new_tokens=255,
                    decoder_input_ids=batch["labels"][:, :4].to("cuda"),
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
    all = input_arg.get("all", False)

    ############
    #  Model   #
    ############

    processor = WhisperProcessor.from_pretrained(
        input_arg["model_config"], task="transcribe", dropout=dropout, language=None
    )
    audio_feature_key = "input_ids"
    if all:
        special_tokens_dict = {'additional_special_tokens': ['<|ast|>', '<|ceb|>', '<|ckb|>', '<|fil|>', '<|ful|>', '<|gle|>', '<|ibo|>', '<|kam|>', '<|kea|>', '<|kir|>', '<|lug|>', '<|luo|>', '<|msa|>', '<|mya|>', '<|nbl|>', '<|nso|>', '<|nya|>', '<|ori|>', '<|orm|>', '<|pan|>', '<|pus|>', '<|sot|>', '<|ssw|>', '<|tsn|>', '<|tso|>', '<|umb|>', '<|ven|>', '<|wol|>', '<|xho|>', '<|zul|>'] + processor.tokenizer.all_special_tokens}
    else:
        special_tokens_dict = {'additional_special_tokens': ['<|new|>'] + processor.tokenizer.all_special_tokens}
    num_added_toks = processor.tokenizer.add_special_tokens(special_tokens_dict)
    

    # load from base model
    model = Whisper_Modified.from_pretrained(input_arg["model_config"])
    config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
    model = get_peft_model(model, config)
    model.resize_token_embeddings(len(processor.tokenizer))

    model = model.to("cuda")
    
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    
    model.print_trainable_parameters()

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, audio_feature_key=audio_feature_key)
    ############
    #  Dataset #
    ############
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
    data_train = data_train.map(encode_dataset, fn_kwargs={"processor": processor, "all": all})
    weight_train = get_weight(processor, model, data_train, all)
    # torch.save(weight_train, "test.pt")
    # weight_train = torch.load("test.pt")
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

    data_test = data_test.map(encode_dataset, fn_kwargs={"processor": processor, "all": all})

    # load from base model
    if all:
        lang_distribution = weight_train
        language_id_tokens = list(model.generation_config.lang_to_id.values())
        language_id_tokens.extend([i for i in range(51865, 51895)])
        for key, value in lang_distribution.items():
            lang_distribution[key] = lang_distribution[key][lang_distribution[key].nonzero(as_tuple=True)].tolist()
        model = Whisper_Modified.from_pretrained(pretrained_model_name_or_path=input_arg["model_config"], new_embedding=lang_distribution, language_tokens=language_id_tokens)
        config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
    else:
        lang_distribution = weight_train.squeeze(0)
        language_id_tokens = lang_distribution.nonzero().squeeze(1)
        lang_distribution = lang_distribution[lang_distribution.nonzero(as_tuple=True)]
        model = Whisper_Modified.from_pretrained(pretrained_model_name_or_path=input_arg["model_config"], new_embedding=lang_distribution.tolist(), language_tokens=language_id_tokens.tolist())
        config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
    model = get_peft_model(model, config)
    model.resize_token_embeddings(len(processor.tokenizer))

    model = model.to("cuda")
    
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    
    model.print_trainable_parameters()
    model = experiment(
        input_arg,
        model,
        processor,
        data_collator,
        data_train,
        data_test,
        time,
        output_dir=input_arg["output_dir"],
        weight=None,
    )

if __name__ == "__main__":
    main()
