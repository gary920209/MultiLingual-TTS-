exp_name="zero_shot_vanilla_plus_all"
for dir in data/ml_superb/sixth_edition/languages/*;do
    size=large-v2
    lang=$(basename $dir)
    output_dir=outputs/$exp_name/$lang
    mkdir -p $output_dir
    python3 vanilla_inference.py \
        --size $size \
        --batch 1 \
        --custom_set_train $dir/train.csv \
        --custom_set_test $dir/test_val.csv \
        --output_dir $output_dir > $output_dir/output.log
done
