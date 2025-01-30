set -e
exp_name="new_token_finetune_ws_utterance_corrected"
for dir in data/ml_superb/sixth_edition/languages/*;do
    size=large-v2
    lang=$(basename $dir)
    output_dir=outputs/$exp_name/$lang
    mkdir -p $output_dir
    if [ $lang == "all" ]; then
        python3 new_token_finetune_ws_inference.py \
            --size $size \
            --batch 1 \
            --grad_accum 8 \
            --epoch 5 \
            --custom_set_train $dir/train.csv \
            --custom_set_test $dir/test_val.csv \
            --all \
            --output_dir $output_dir > $output_dir/output.log
    else
        python3 new_token_finetune_ws_inference.py \
        --size $size \
        --batch 1 \
        --grad_accum 8 \
        --epoch 5 \
        --custom_set_train $dir/train.csv \
        --custom_set_test $dir/test_val.csv \
        --output_dir $output_dir > $output_dir/output.log
    fi
done

exp_name="new_token_finetune_ws_corpus_corrected"
for dir in data/ml_superb/sixth_edition/languages/*;do
    size=large-v2
    lang=$(basename $dir)
    output_dir=outputs/$exp_name/$lang
    mkdir -p $output_dir
    if [ $lang == "all" ]; then
        python3 new_token_finetune_ws_inference.py \
            --size $size \
            --batch 1 \
            --grad_accum 8 \
            --epoch 5 \
            --custom_set_train $dir/train.csv \
            --custom_set_test $dir/test_val.csv \
            --all \
            --corpus_wise \
            --output_dir $output_dir > $output_dir/output.log
    else
        python3 new_token_finetune_ws_inference.py \
            --size $size \
            --batch 1 \
            --grad_accum 8 \
            --epoch 5 \
            --custom_set_train $dir/train.csv \
            --custom_set_test $dir/test_val.csv \
            --corpus_wise \
            --output_dir $output_dir > $output_dir/output.log
    fi
done
