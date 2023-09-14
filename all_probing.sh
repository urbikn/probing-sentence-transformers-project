#!/bin/bash

########## 
# This script is used to run all probing tasks for all models.
#
# It additionally implements the function of running N experiments in parallel.
##########

# Define parameters
# If seed is not specified, it will be set to 12, to have a way to run multiple times
if [ -z "$1" ]; then
    seed=12
else
    seed=$1
fi
batch_size=32

# Define an array of tasks
tasks=("bigram_shift" "obj_number" "sentence_length" "top_constituents" "word_content" "coordination_inversion" "odd_man_out" "subj_number" "tree_depth" "past_present")

# Define an array of layers
layers=(0 1 2 3 4 5 6 7 8 9 10 11 12)
layers=(12)

# Define an array of models
models=("mpnet-base-v2" "MiniLM-L12-v2")
counter_limits=(12 12)


# Iterate over all models 
for i in "${!models[@]}"; do
    model=${models[$i]}
    counter_limit=${counter_limits[$i]}

    echo ""
    echo "=========================="
    echo "Starting $model Probing Tasks"
    echo "=========================="
    echo ""

    counter=0
    # Iterate over all tasks
    for task in "${tasks[@]}"; do
        echo "= Task $task"

        # Iterate over all layers
        for layer in "${layers[@]}"; do
            echo "Layer: $layer"
            python3 src/main_probing.py --model $model --task $task --seed $seed --batch_size $batch_size --training_data .embeddings/$task.txt --embedding_data .embeddings/$model-layer-$layer.$task.pca.pt --training_id $model-layer-$layer-pca-$seed &
            counter=$((counter+1))
            if [ $counter -eq $counter_limit ]; then
                wait  # Wait for all $counter_limit scripts to complete
                counter=0
            fi
        done

        if [ $counter -eq $counter_limit ]; then
            wait  # Wait for all $counter_limit scripts to complete
            counter=0
        fi
    done

    wait
done

# Now run the same experiments but for only the baseline model
model="bilstm"
counter_limit=12

echo ""
echo "=========================="
echo "Starting $model Probing Tasks"
echo "=========================="
echo ""

counter=0
# Iterate over all tasks
for task in "${tasks[@]}"; do

    echo "= Task $task"
    python3 src/main_probing.py --model $model --task $task --seed $seed --batch_size $batch_size --training_data .embeddings/$task.txt --embedding_data .embeddings/$model.$task.pca.pt  --training_id $model-pca-$seed &
    counter=$((counter+1))
    if [ $counter -eq $counter_limit ]; then
        wait  # Wait for all $counter_limit scripts to complete
        counter=0
    fi
done
wait