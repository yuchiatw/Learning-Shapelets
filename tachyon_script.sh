#!/bin/bash

# Array of datasets
features=(
    'cpu_system'
    'boottime'
    'Pool Size Time_P1'
    'mem_free'
    'Missed Buffers_P1'
    'bytes_out'
    'cpu_user'
    'cpu_idle'
    'Pool Size Data_P1'
    'pkts_out'  
    'load_fifteen'
    'part_max_used'
    'load_five'
    'load_one'
)

# Path to the exp_functions.py script
exp_script="python tachyon_exp.py"

for feature in "${features[@]}"; do
    $exp_script --feature "$feature"
done