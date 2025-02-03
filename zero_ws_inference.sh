# utterance wise
exp_name="zero_shot_utterance_wise_plus_all"
dir=data/mlsuperb2
size=large-v2
output_dir=outputs/$exp_name
mkdir -p $output_dir
python3 zero_ws_inference.py \
    --size $size \
    --batch 1 \
    --lang_list languages.json \
    --custom_set_test data/test_30.csv \
    --output_dir $output_dir > $output_dir/output.log
