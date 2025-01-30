# Weghted Sum Embedding as Initialization for Inference and Finetuning of Whisper

## Installation
Use the exact version of the packages in the ``requirements.txt``.
```
pip install -r requirements.txt
```

## Data Format
The data format follows ml_superb. Each instance contains the path to the wav file and the text.
For example,
```
path,text
/tmp2/gordonzz/Whisper_Experiments/data/ml_superb/sixth_edition/fleurs/ast/wav/fleurs_ast_000067.wav,EN CUANTES A XAPóN XAPóN YERA UN PAíS-ISLLA IGUAL QUE GRAN BRETAñA
/tmp2/gordonzz/Whisper_Experiments/data/ml_superb/sixth_edition/fleurs/ast/wav/fleurs_ast_000068.wav,DE FRACASAR LOS ALIAOS YE PROBABLE QU'ALEMAñA CONQUISTARE GRAN BRETAñA Y EL RESTU D'EUROPA
/tmp2/gordonzz/Whisper_Experiments/data/ml_superb/sixth_edition/fleurs/ast/wav/fleurs_ast_000069.wav,LES IMáXENES D’INFRARROXU AMUESEN QUE LES VARIACIONES DE TEMPERATURA ENTE’L DíA Y LA NUECHE PRUEBEN QUE YE FáCIL QUE SEYAN CUEVES
...
```

If you need the whisper-unseen version of ml_superb, please contact me.
## Inference
Use the script ``ws_inference.py``. The arguments are:
- batch: batch size.
- custom_test_set: the csv path of the testing data.
- size: the Whisper version (``large-v2``, ``large-v3``)
- corpus_wise: All the weighted sum embeddings will be averaged to obtain a shared embedding.
- output_dir: the location to save the output log (containing CER and WER results) and prediction.

Refer to ``ws_inference.sh`` for my usage.
## Finetune
There are two settings.
### Fix Embedding Layer
Use the script ``ws_finetune_untrainable.py``. The arguments are:
- grad_accm: gradient accumulation.
- epoch: training epoch (currently I have not implemented earlystopping)
- custom_train_set: the csv path of the training data.

Refer to ``ws_finetune_untrainable.sh`` for my usage.
### Trainable Embedding Layer
Use the script ``ws_finetune_untrainable.py``.
The embedding layer and the weight is trainable in this case. The weight is initialized by the corpus-wise averaged weight, so the default behavior is ``corpus-wise`` here.
Other arguments are the same with the former ones.

Refer to ``ws_finetune_trainable.sh`` for my usage.

## TODO
- [ ] Try this with different seeds.
- [ ] Add vanilla inference script.
