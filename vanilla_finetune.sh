# set -e
# exp_name="new_token_finetune_vanilla_inference"
# for dir in data/ml_superb/sixth_edition/languages/ast;do
#     size=large-v2
#     lang=$(basename $dir)
#     if [ $lang == "all" ]; then
#         continue
#     fi
#     output_dir=outputs/$exp_name/$lang
#     mkdir -p $output_dir
#     python3 vanilla_finetune.py \
#         --size $size \
#         --batch 1 \
#         --grad_accum 8 \
#         --epoch 5 \
#         --custom_set_train $dir/train.csv \
#         --custom_set_test $dir/test_val.csv \
#         --output_dir $output_dir > $output_dir/output.log
# done

#run all language at once
set -e
exp_name="new_token_finetune_vanilla_inference"
for dir in data/ml_superb/sixth_edition/languages/all;do
    size=large-v2
    lang=$(basename $dir)
    output_dir=outputs/$exp_name/$lang
    mkdir -p $output_dir
    python3 vanilla_finetune.py \
        --size $size \
        --batch 1 \
        --grad_accum 8 \
        --epoch 5 \
        --custom_set_train $dir/train.csv \
        --custom_set_test $dir/test_val.csv \
        --output_dir $output_dir > $output_dir/output.log
done