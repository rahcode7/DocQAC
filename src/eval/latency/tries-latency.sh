#!/bin/bash

# Name of the Python script
#which python3

#source activate autosuggest
#export PYTHONPATH=/opt/homebrew/bin/python3


# RUNS=10
FILE_LIST=('test_formatted_seen_query-seen_doc_test.tsv' 'test_formatted_unseen_query-seen_doc_test.tsv' 'test_formatted_unseen_query-unseen_doc_test.tsv' 'test_formatted_seen_query-unseen_doc_test.tsv' )


# Measure the time taken to run the Python script
echo "Running" # $PYTHON_SCRIPT..."

############ GLOBAL TRIES
MODEL_TYPE='global-tries'
RUNS=1
mkdir $MODEL_TYPE
PYTHON_SCRIPT="query-auto-suggest/src/code/mpc/parallel_mpc_inference.py"
rm -f $MODEL_TYPE/latency.tsv
for FILE in "${FILE_LIST[@]}"; do
    # RUN EACH 10 TIMES 
    for ((i=1;i<=$RUNS;i++)); do 
        #{time python3 "$PYTHON_SCRIPT" --input_file datasets/inputs/"$FILE"  --main_trie datasets/outputs/docquery-tries --output_file datasets/results/docq-tries/completions_$FILE.mpc}  2> $MODEL_TYPE/$FILE.txt 2> /dev/null
        start_time=$(gdate +%s%3N)
        python3 "$PYTHON_SCRIPT" --input_file datasets/inputs/"$FILE"  --main_trie datasets/outputs/global-tries/main.mpc --suffix_trie datasets/outputs/global-tries/suffix.mpc --output_file datasets/results/docq-tries/completions_$FILE.mpc
        end_time=$(gdate +%s%3N)
        execution_time=$((end_time - start_time))
        echo "Execution time: $execution_time milliseconds"
        echo -e "$i\t$MODEL_TYPE\t$FILE\t$execution_time" >> $MODEL_TYPE/latency.tsv
    done
done


############ DOCQUERY TRIES
# MODEL_TYPE='docquery-tries'
# mkdir $MODEL_TYPE
# rm -f $MODEL_TYPE/latency.tsv
# PYTHON_SCRIPT="query-auto-suggest/src/code/mpc/parallel_mpc_inference-docq.py"
# FILE_LIST=('test_formatted_seen_query-seen_doc_test.tsv' 'test_formatted_unseen_query-seen_doc_test.tsv')
# RUNS=10
# for FILE in "${FILE_LIST[@]}"; do
#     # RUN EACH 10 TIMES 
#     for ((i=1;i<=$RUNS;i++)); do 
#         #{time python3 "$PYTHON_SCRIPT" --input_file datasets/inputs/"$FILE"  --main_trie datasets/outputs/docquery-tries --output_file datasets/results/docq-tries/completions_$FILE.mpc}  2> $MODEL_TYPE/$FILE.txt 2> /dev/null
#         start_time=$(gdate +%s%3N)
#         python3 "$PYTHON_SCRIPT" --input_file datasets/inputs/"$FILE"  --main_trie datasets/outputs/docquery-tries  --output_file datasets/results/docq-tries/completions_$FILE.mpc
#         end_time=$(gdate +%s%3N)
#         execution_time=$((end_time - start_time))
#         echo "Execution time: $execution_time milliseconds"
#         echo -e "$i\t$MODEL_TYPE\t$FILE\t$execution_time" >> $MODEL_TYPE/latency.tsv
#     done
# done



# ############ DOC TRIES  - doc
# MODEL_TYPE='doc-tries'
# mkdir $MODEL_TYPE
# rm -f $MODEL_TYPE/latency.tsv
# PYTHON_SCRIPT="query-auto-suggest/src/code/mpc/parallel_mpc_inference-doc.py"
# FILE_LIST=('test_formatted_seen_query-seen_doc_test.tsv' 'test_formatted_unseen_query-seen_doc_test.tsv')
# RUNS=1
# for FILE in "${FILE_LIST[@]}"; do
#     # RUN EACH 10 TIMES 
#     for ((i=1;i<=$RUNS;i++)); do 
#         #{time python3 "$PYTHON_SCRIPT" --input_file datasets/inputs/"$FILE"  --main_trie datasets/outputs/docquery-tries --output_file datasets/results/docq-tries/completions_$FILE.mpc}  2> $MODEL_TYPE/$FILE.txt 2> /dev/null
#         start_time=$(gdate +%s%3N)
#         python3 "$PYTHON_SCRIPT" --input_file datasets/inputs/"$FILE"  --main_trie datasets/outputs/docngram-tries  --output_file datasets/results/doc-tries/completions_$FILE.mpc
#         end_time=$(gdate +%s%3N)
#         execution_time=$((end_time - start_time))
#         echo "Execution time: $execution_time milliseconds"
#         echo -e "$i\t$MODEL_TYPE\t$FILE\t$execution_time" >> $MODEL_TYPE/latency.tsv
#     done
# done


# ############ DOC TRIES - unseen doc
# MODEL_TYPE='doc-tries'
# mkdir $MODEL_TYPE
# #rm -f $MODEL_TYPE/latency.tsv
# PYTHON_SCRIPT="query-auto-suggest/src/code/mpc/parallel_mpc_inference-doc.py"
# FILE_LIST=('test_formatted_unseen_query-unseen_doc_test.tsv')
# RUNS=1
# for FILE in "${FILE_LIST[@]}"; do
#     # RUN EACH 10 TIMES 
#     for ((i=1;i<=$RUNS;i++)); do 
#         #{time python3 "$PYTHON_SCRIPT" --input_file datasets/inputs/"$FILE"  --main_trie datasets/outputs/docquery-tries --output_file datasets/results/docq-tries/completions_$FILE.mpc}  2> $MODEL_TYPE/$FILE.txt 2> /dev/null
#         start_time=$(gdate +%s%3N)
#         python3 "$PYTHON_SCRIPT" --input_file datasets/inputs/"$FILE"  --main_trie datasets/outputs/docngram-tries-test-unseenq-ud  --output_file datasets/results/doc-tries/completions_$FILE.mpc
#         end_time=$(gdate +%s%3N)
#         execution_time=$((end_time - start_time))
#         echo "Execution time: $execution_time milliseconds"
#         echo -e "$i\t$MODEL_TYPE\t$FILE\t$execution_time" >> $MODEL_TYPE/latency.tsv
#     done
# done


# ############ DOC TRIES - unseen doc
# MODEL_TYPE='doc-tries'
# #mkdir $MODEL_TYPE
# #rm -f $MODEL_TYPE/latency.tsv
# PYTHON_SCRIPT="query-auto-suggest/src/code/mpc/parallel_mpc_inference-doc.py"
# FILE_LIST=('test_formatted_seen_query-unseen_doc_test.tsv')
# RUNS=1
# for FILE in "${FILE_LIST[@]}"; do
#     # RUN EACH 10 TIMES 
#     for ((i=1;i<=$RUNS;i++)); do 
#         #{time python3 "$PYTHON_SCRIPT" --input_file datasets/inputs/"$FILE"  --main_trie datasets/outputs/docquery-tries --output_file datasets/results/docq-tries/completions_$FILE.mpc}  2> $MODEL_TYPE/$FILE.txt 2> /dev/null
#         start_time=$(gdate +%s%3N)
#         python3 "$PYTHON_SCRIPT" --input_file datasets/inputs/"$FILE"  --main_trie datasets/outputs/docngram-tries-test-seenq-ud  --output_file datasets/results/doc-tries/completions_$FILE.mpc
#         end_time=$(gdate +%s%3N)
#         execution_time=$((end_time - start_time))
#         echo "Execution time: $execution_time milliseconds"
#         echo -e "$i\t$MODEL_TYPE\t$FILE\t$execution_time" >> $MODEL_TYPE/latency.tsv
#     done
# done

# Instructions - 
# Download https://drive.google.com/drive/folders/1GLboNumJ6xrZmuB1PsRelF7ixhogr391?usp=drive_link
# cd tries
# pip install -r requirements.txt
# unzip datasets/outputs.zip datasets
# source query-auto-suggest/src/code/mpc/latency.sh
# Output generated in 3 new folders doc-tries,global-tries,docquery-tries
