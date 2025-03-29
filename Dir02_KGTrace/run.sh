#!/bin/bash

. $1
export PYTHONPATH="."
gpu_id=1
txt_dir="datasets/data_preprocessed/FB15K-237/toy_data/"
bern_txt_dir="datasets/data_preprocessed/FB15K-237/toy_data/toy_data_transD"
total_iterations=10000
path_length=3
hidden_size=50
embedding_size=50
batch_size=512
beta=0.14
Lambda=0.14
use_cluster_embeddings=0
use_entity_embeddings=0
train_entity_embeddings=0
train_relation_embeddings=1
base_output_dir="output/fb15k-237/test1"
load_model=0
model_load_dir="saved_models/fb15k-237"
nell_evaluation=0
learning_rate=1e-3
gamma=0.1
cmd="python3 trainer.py --learning_rate $learning_rate --base_output_dir $base_output_dir --path_length $path_length --hidden_size $hidden_size --embedding_size $embedding_size \
    --batch_size $batch_size --beta $beta --Lambda $Lambda --use_entity_embeddings $use_entity_embeddings --use_cluster_embeddings $use_cluster_embeddings\
    --train_entity_embeddings $train_entity_embeddings --train_relation_embeddings $train_relation_embeddings \
    --txt_dir $txt_dir --bern_txt_dir $bern_txt_dir --model_load_dir $model_load_dir --load_model $load_model --total_iterations $total_iterations\
    --nell_evaluation $nell_evaluation --learning_rate $learning_rate --gamma $gamma"



echo "Executing $cmd"

CUDA_VISIBLE_DEVICES=$gpu_id $cmd

