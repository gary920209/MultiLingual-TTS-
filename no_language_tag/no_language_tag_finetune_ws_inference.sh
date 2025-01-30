set -e
exp_name="no_language_id_finetune_ws_utterance"
for dir in data/ml_superb/sixth_edition/languages/luo;do
    size=large-v2
    lang=$(basename $dir)
    if [ $lang == "all" ]; then
        continue
    fi
    output_dir=outputs/$exp_name/$lang
    mkdir -p $output_dir
    python3 no_language_tag_finetune_ws_inference.py \
        --size $size \
        --batch 1 \
        --grad_accum 8 \
        --epoch 5 \
        --custom_set_train $dir/train.csv \
        --custom_set_test $dir/test_val.csv \
        --output_dir $output_dir > $output_dir/output.log
done
exp_name="no_language_id_finetune_ws_corpus"
for dir in data/ml_superb/sixth_edition/languages/*;do
    size=large-v2
    lang=$(basename $dir)
    if [ $lang == "all" ]; then
        continue
    fi
    output_dir=outputs/$exp_name/$lang
    mkdir -p $output_dir
    python3 no_language_tag_finetune_ws_inference.py \
        --size $size \
        --batch 1 \
        --grad_accum 8 \
        --epoch 5 \
        --custom_set_train $dir/train.csv \
        --custom_set_test $dir/test_val.csv \
        --corpus_wise \
        --output_dir $output_dir > $output_dir/output.log
done
