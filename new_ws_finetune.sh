set -e
exp_name="fine_tune_trainable_with_new_token_corrected"
size=large-v2
dir=data
output_dir=outputs/$exp_name/all
python3 ws_finetune_trainable.py \
    --size $size \
    --batch 1 \
    --grad_accum 8 \
    --epoch 5 \
    --custom_set_train $dir/train.csv \
    --custom_set_test $dir/test_val.csv \
    --all \
    --output_dir $output_dir > $output_dir/output.log