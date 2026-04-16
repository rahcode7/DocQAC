#!/bin/bash


dtypes=("test_formatted_seen_query-seen_doc_test" "test_formatted_seen_query-unseen_doc_test" "test_formatted_unseen_query-seen_doc_test" "test_formatted_unseen_query-unseen_doc_test")
rag_types=("rag_sparse" "rag_dense" "rag_sim_doc")

# for dtype in "${dtypes[@]}"
# do
#     echo " - Processing ${dtype} ..."
#     for rag_type in "${rag_types[@]}"
#     do
#         echo " -- Processing ${rag_type} ..."
#         python prepare_rag_data.py --raw_doc_file ./Document-AS/raw_dataset/docs \
#                         --input_file ./Document-AS/processed_test_dataset/rag/${dtype}.tsv \
#                         --context_type ${rag_type} \
#                         --max_limit -1 \
#                         --num_worker 8  \
#                         --output_file ./Document-AS/processed_test_dataset/rag/${dtype}.tsv
#     done 
# done


for dtype in "${dtypes[@]}"
do
    echo " - Processing ${dtype} ..."
    for rag_type in "${rag_types[@]}"
    do
        echo " -- Processing ${rag_type} ..."
        python prepare_rag_data.py --raw_doc_file datasets/master/docs \
                        --input_file datasets/processed_test_dataset/rag/${dtype}.tsv \
                        --context_type ${rag_type} \
                        --max_limit -1 \
                        --num_worker 1  \
                        --output_file datasets/processed_test_dataset/rag/${dtype}.tsv
    done 
done