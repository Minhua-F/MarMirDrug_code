import os

data_input_dir="datasets/data_preprocessed/FB15K-237/toy_data/"
vocab_dir="datasets/data_preprocessed/FB15K-237/toy_data/toy_data_transD/vocab"
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
base_output_dir="output/fb15k-237"
load_model=0
model_load_dir="saved_models/fb15k-237"
nell_evaluation=0
learning_rate=1e-3
gamma=0.1

os.system('python3 trainer.py --learning_rate $' + str(learning_rate) + ' --base_output_dir $' + base_output_dir + ' --path_length $' + str(path_length) +
          ' --hidden_size $' + str(hidden_size) + ' --embedding_size $' + str(embedding_size) + ' --batch_size $' + str(batch_size) + ' --beta $' + str(beta) +
          ' --Lambda $' + str(Lambda) + ' --use_entity_embeddings $' + str(use_entity_embeddings) + ' --use_cluster_embeddings $' + str(use_cluster_embeddings) +
          ' --train_entity_embeddings $' + str(train_entity_embeddings) + ' --train_relation_embeddings $' + str(train_relation_embeddings) +
          ' --data_input_dir $' + data_input_dir + ' --vocab_dir $' + vocab_dir + ' --model_load_dir $' + model_load_dir +
          ' --load_model $' + str(load_model) + ' --total_iterations $' + str(total_iterations) + ' --nell_evaluation $' + str(nell_evaluation) +
          ' --learning_rate $' + str(learning_rate) + ' --gamma $' + str(gamma))
