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

NEW_TOKEN_TO_ID = {'ast': 51865, 'ceb': 51866, 'ckb': 51867, 'fil': 51868, 'ful': 51869, 'gle': 51870, 'ibo': 51871, 'kam': 51872, 'kea': 51873, 'kir': 51874, 'lug': 51875, 'luo': 51876, 'msa': 51877, 'mya': 51878, 'nbl': 51879, 'nso': 51880, 'nya': 51881, 'ori': 51882, 'orm': 51883, 'pan': 51884, 'pus': 51885, 'sot': 51886, 'ssw': 51887, 'tsn': 51888, 'tso': 51889, 'umb': 51890, 'ven': 51891, 'wol': 51892, 'xho': 51893, 'zul': 51894}

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
    lang=batch["path"].split("/")[-3]
    lang_id=NEW_TOKEN_TO_ID[lang]
    batch["labels"] = batch["labels"][:1] + [lang_id] + batch["labels"][1:]
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

def experiment(input_arg, model, processor, data_collator, data_train, data_test, time, output_dir, eval_only):
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

    only_eval = input_arg.get("only_eval", False)
    ############
    #  Model   #
    ############

    processor = WhisperProcessor.from_pretrained(
        input_arg["model_config"], task="transcribe", dropout=dropout, language=None
    )
    audio_feature_key = "input_ids"
    special_tokens_dict = {'additional_special_tokens': ['<|ast|>', '<|ceb|>', '<|ckb|>', '<|fil|>', '<|ful|>', '<|gle|>', '<|ibo|>', '<|kam|>', '<|kea|>', '<|kir|>', '<|lug|>', '<|luo|>', '<|msa|>', '<|mya|>', '<|nbl|>', '<|nso|>', '<|nya|>', '<|ori|>', '<|orm|>', '<|pan|>', '<|pus|>', '<|sot|>', '<|ssw|>', '<|tsn|>', '<|tso|>', '<|umb|>', '<|ven|>', '<|wol|>', '<|xho|>', '<|zul|>'] + processor.tokenizer.all_special_tokens}
    num_added_toks = processor.tokenizer.add_special_tokens(special_tokens_dict)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, audio_feature_key=audio_feature_key)

    # load from base model
    model = WhisperForConditionalGeneration.from_pretrained(input_arg["model_config"])
    config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
    model = get_peft_model(model, config)
    model.resize_token_embeddings(len(processor.tokenizer))   
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
    )

if __name__ == "__main__":
    main()
