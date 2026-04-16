#!/bin/bash

# generative config
gen_token=40
batch_size=4
beam_size=10
context_type=1 # pass context type value from 1, 2, 3 and 4
TRIE_PATH="$1"

#path to evaluation directory
eval_dir=./test_dataset

# please pass the peft checkpoint of trained model
peft_checkpoint_path=${path to model checkpoint}

eval_files=("test_formatted_seen_query-seen_doc_test" "test_formatted_unseen_query-seen_doc_test" "test_formatted_seen_query-unseen_doc_test" "test_formatted_unseen_query-unseen_doc_test")


for eval_file in "${eval_files[@]}"
do
    echo "====== Generating for ${eval_file} ======"
    output_dir=./model_outputs
    log_dir=./gen_logs
    mkdir -p ${output_dir}
    mkdir -p ${log_dir}
    rm -f ${log_dir}/${eval_file}.log;python generator.py \
        --model_path ${peft_checkpoint_path} \
        --input_file ${eval_dir}/${eval_file}.tsv \
        --output_file ${output_dir}/${eval_file}.jsonl \
        --num_workers 8 \
        --beam_size ${beam_size} \
        --num_seq ${beam_size} \
        --max_limit -1 \
        --context_type ${context_type} \
        --max_src_token 512 --max_gen_token ${gen_token} \
        --batch_size ${batch_size} 2>&1 | tee -a ${log_dir}/${eval_file}.log \
        --trie_path $TRIE_PATH
        --alpha 0.1 \
        --beta 0.0 
done
