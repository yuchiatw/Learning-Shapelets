#!/bin/bash

# Array of datasets
model=('LS_Transformer' 'JOINT' 'BOSS' 'vanilla')

# Path to the exp_functions.py script
exp_script="python exp_functions.py"

for model_name in "${model[@]}"; do
    $exp_script --model "$model_name"
done
