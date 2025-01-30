set -e
# utterance wise
exp_name="finetune_fix_embed"
for dir in data/ml_superb/sixth_edition/languages/*;do
    size=large-v2
    lang=$(basename $dir)
    output_dir=outputs/$exp_name/$lang
    mkdir -p $output_dir
    if [ $lang == "all" ]; then
        python3 ws_finetune_untrainable.py \
            --size $size \
            --batch 1 \
            --grad_accum 8 \
            --epoch 5 \
            --all \
            --custom_set_train $dir/train.csv \
            --custom_set_test $dir/test_val.csv \
            --output_dir $output_dir > $output_dir/output.log
    else
        continue
        # python3 ws_finetune_untrainable.py \
        #     --size $size \
        #     --batch 1 \
        #     --grad_accum 8 \
        #     --epoch 5 \
        #     --custom_set_train $dir/train.csv \
        #     --custom_set_test $dir/test_val.csv \
        #     --output_dir $output_dir > $output_dir/output.log
    fi
done

# corpus wise
exp_name="finetune_fix_embed_corpus_wise"
for dir in data/ml_superb/sixth_edition/languages/*;do
    size=large-v2
    lang=$(basename $dir)
    output_dir=outputs/$exp_name/$lang
    mkdir -p $output_dir
    if [ $lang == "all" ]; then
        python3 ws_finetune_untrainable.py \
            --size $size \
            --batch 1 \
            --grad_accum 8 \
            --epoch 5 \
            --all \
            --corpus_wise \
            --custom_set_train $dir/train.csv \
            --custom_set_test $dir/test_val.csv \
            --output_dir $output_dir > $output_dir/output.log
    else
        continue
        # python3 ws_finetune_untrainable.py \
        #     --size $size \
        #     --batch 1 \
        #     --grad_accum 8 \
        #     --epoch 5 \
        #     --corpus_wise \
        #     --custom_set_train $dir/train.csv \
        #     --custom_set_test $dir/test_val.csv \
        #     --output_dir $output_dir > $output_dir/output.log
    fi
done
