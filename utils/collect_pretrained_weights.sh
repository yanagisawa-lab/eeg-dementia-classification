#!/bin/bash

# Define base path
BASE_PATH="train/MNet_1000_seg/submission_2023_0614"

# Iterate over subtype comparison directories
for subtype in $(ls $BASE_PATH); do
    # Ensure the directory exists under the current working directory
    mkdir -p "pretrained_weights/$subtype"

    # Find and copy the desired files
    find "$BASE_PATH/$subtype" -name "model_*.pth" -exec cp {} "pretrained_weights/$subtype/" \;
done
