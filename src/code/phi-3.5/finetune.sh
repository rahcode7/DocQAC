#!/bin/bash

batch_size=$1 #128
model_name=phi-3.5-mini-instruct
epochs=5
grad_accum=1
max_seq_len=512
dataset_path={path_to_dataset} # please change the path to dataset

base_dir=./trained_model
context_types=("prefix_only" "title_doc" "title_url" "title_url_doc" "title_url_summary" "title_url_yake" "title_url_rag_sparse" "title_url_rag_dense" "title_url_rag_sim_doc")

for ctx in "${context_types[@]}"
do
    full_model_name=${model_name}_bs${batch_size}_ep${epochs}_ctx${ctx}
    model_dir=${base_dir}/${full_model_name}
    mkdir -p ${base_dir}

    log_dir=${model_dir}/logs
    mkdir -p $log_dir

    rm -f ${log_dir}/${full_model_name}.log;accelerate launch --multi_gpu --num_processes 8 phi_35_script.py \
        --batch_size ${batch_size} \
        --epochs ${epochs} \
        --data_path ${dataset_path} \
        --train_data_count -1 --val_data_count -1 \
        --context_type ${ctx} \
        --max_seq_len ${max_seq_len} \
        --checkpoint_path ${model_dir}/peft_checkpoints \
        --grad_accum ${grad_accum} 2>&1 | tee -a ${log_dir}/${full_model_name}.log
done