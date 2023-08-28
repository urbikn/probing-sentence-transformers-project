#!/bin/bash

########## 
# This script is used to run all probing tasks for all models.
#
# It additionally implements the function of running N experiments in parallel.
##########

# Define parameters
seed=12
batch_size=32

# Define an array of tasks
tasks=("bigram_shift" "obj_number" "sentence_length" "top_constituents" "word_content" "coordination_inversion" "odd_man_out" "subj_number" "tree_depth")

# Define an array of layers
layers=(0 1 2 3 4 5 6 7 8 9 10 11 12)

# Define an array of models
models=("mpnet" "MiniLM-L12-v2")

# Iterate over all models
for model in "${models[@]}"; do
    echo ""
    echo "=========================="
    echo "Starting $model Probing Tasks"
    echo "=========================="
    echo ""

    # Iterate over all tasks
    for task in "${tasks[@]}"; do
        counter=0

        # Iterate over all layers
        for layer in "${layers[@]}"; do
            python3 src/main_probing.py --model $model --task $task --seed $seed --batch_size $batch_size --training_data .embeddings/$task.txt --embedding_data .embeddings/$model-layer-$layer.$task.pt &
            counter=$((counter+1))
            if [ $counter -eq 4 ]; then
                wait  # Wait for 4 scripts to complete
                counter=0
            fi
        done
        wait  # Wait for any remaining scripts to complete
    done
done
