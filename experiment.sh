#!/bin/bash
size="large-v3"
base_dir="data/mlsuperb2"
exp_name=$1
test_set=$2
checkpoint_dir="checkpoint"/$exp_name
output_dir="outputs/inference"/$exp_name

# Iterate through all checkpoint folders under checkpoint_dir, with the name format "checkpoint-xxxxx"
for checkpoint in $checkpoint_dir/checkpoint-*; do
    # Extract the step number from the checkpoint folder name
    step=$(basename $checkpoint | cut -d'-' -f2)
    
    # Create output directory
    mkdir -p $output_dir
    
    # Create a specific output directory for this checkpoint
    checkpoint_output_dir="$output_dir/step_$step"
    mkdir -p $checkpoint_output_dir
    
    echo "Processing checkpoint: $checkpoint"
    python3 zero_ws_inference.py \
        --size $size \
        --batch 1 \
        --lang_list languages.json \
        --base_dir $base_dir \
        --custom_set_test $test_set \
        --output_dir $checkpoint_output_dir \
        --checkpoint $checkpoint \
        > $checkpoint_output_dir/output.log
done
