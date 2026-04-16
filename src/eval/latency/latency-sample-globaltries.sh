#!/bin/bash

# Name of the Python script
#which python3

#source activate autosuggest
#export PYTHONPATH=/opt/homebrew/bin/python3


# RUNS=10
FILE_LIST=('1000_sample_test.tsv')

# Measure the time taken to run the Python script
echo "Running" # $PYTHON_SCRIPT..."

############################################################## GLOBAL TRIES
#### Prefix 
MODEL_TYPE='global-tries'
EXP_NAME='prefix'
echo "running $MODEL_TYPE - $EXP_NAME ......"
RUNS=1
mkdir $MODEL_TYPE
mkdir datasets/results/$MODEL_TYPE-sample
mkdir $MODEL_TYPE/$EXP_NAME
PYTHON_SCRIPT="query-auto-suggest/src/code/mpc/parallel_mpc_inference.py"
rm -f $MODEL_TYPE/$EXP_NAME/latency.tsv
start_time=$(gdate +%s%3N)
python3 "$PYTHON_SCRIPT" --input_file datasets/inputs/"$FILE"  --main_trie datasets/outputs/global-tries/main.mpc --suffix_trie datasets/outputs/global-tries/suffix.mpc --output_file datasets/results/global-tries-sample/completions_1000_sample_test.mpc --k_completions 10  --exp_name $EXP_NAME
#python3 query-auto-suggest/src/code/mpc/00-reranker-scores.py --completions_path datasets/results/global-tries-sample  --op_path datasets/results/global-tries-sample --exp_type tries --context title_url  --exp_name $EXP_NAME
#python3 query-auto-suggest/src/code/mpc/01-reranker-algo.py --ip_path datasets/results/global-tries-sample --exp_type tries 
end_time=$(gdate +%s%3N)
execution_time=$((end_time - start_time))
echo "Execution time: $execution_time milliseconds"
echo -e "$i\t$MODEL_TYPE\t$FILE\t$execution_time" >> $MODEL_TYPE/$EXP_NAME/latency.tsv



##### Title + URL 
MODEL_TYPE='global-tries'
EXP_NAME='title-url'
echo "running $MODEL_TYPE - $EXP_NAME ......"
RUNS=1
mkdir $MODEL_TYPE
mkdir datasets/results/$MODEL_TYPE-sample
mkdir $MODEL_TYPE/$EXP_NAME
PYTHON_SCRIPT="query-auto-suggest/src/code/mpc/parallel_mpc_inference.py"
rm -f $MODEL_TYPE/$EXP_NAME/latency.tsv
start_time=$(gdate +%s%3N)
python3 "$PYTHON_SCRIPT" --input_file datasets/inputs/"$FILE"  --main_trie datasets/outputs/global-tries/main.mpc --suffix_trie datasets/outputs/global-tries/suffix.mpc --output_file datasets/results/global-tries-sample/completions_1000_sample_test.mpc --k_completions 100  --exp_name $EXP_NAME
python3 query-auto-suggest/src/code/mpc/00-reranker-scores.py --completions_path datasets/results/global-tries-sample  --op_path datasets/results/global-tries-sample --exp_type tries --context title_url  --exp_name $EXP_NAME
python3 query-auto-suggest/src/code/mpc/01-reranker-algo.py --ip_path datasets/results/global-tries-sample --exp_type tries 
end_time=$(gdate +%s%3N)
execution_time=$((end_time - start_time))
echo "Execution time: $execution_time milliseconds"
echo -e "$i\t$MODEL_TYPE\t$FILE\t$execution_time" >> $MODEL_TYPE/$EXP_NAME/latency.tsv


# Title + URL + Doc
MODEL_TYPE='global-tries'
EXP_NAME='title-url-doc'
echo "running $MODEL_TYPE - $EXP_NAME ......"
RUNS=1
mkdir $MODEL_TYPE
mkdir datasets/results/$MODEL_TYPE-sample
mkdir $MODEL_TYPE/$EXP_NAME
PYTHON_SCRIPT="query-auto-suggest/src/code/mpc/parallel_mpc_inference.py"
rm -f $MODEL_TYPE/$EXP_NAME/latency.tsv
start_time=$(gdate +%s%3N)
python3 "$PYTHON_SCRIPT" --input_file datasets/inputs/"$FILE"  --main_trie datasets/outputs/global-tries/main.mpc --suffix_trie datasets/outputs/global-tries/suffix.mpc --output_file datasets/results/global-tries-sample/completions_1000_sample_test.mpc --k_completions 100  --exp_name $EXP_NAME
python3 query-auto-suggest/src/code/mpc/00-reranker-scores.py --completions_path datasets/results/global-tries-sample  --op_path datasets/results/global-tries-sample --exp_type tries --context doc  --exp_name $EXP_NAME
python3 query-auto-suggest/src/code/mpc/01-reranker-algo.py --ip_path datasets/results/global-tries-sample --exp_type tries 
end_time=$(gdate +%s%3N)
execution_time=$((end_time - start_time))
echo "Execution time: $execution_time milliseconds"
echo -e "$i\t$MODEL_TYPE\t$FILE\t$execution_time" >> $MODEL_TYPE/$EXP_NAME/latency.tsv


# Title + URL + Summary
MODEL_TYPE='global-tries'
EXP_NAME='title-url-summary'
echo "running $MODEL_TYPE - $EXP_NAME ......"
RUNS=1
mkdir $MODEL_TYPE
mkdir datasets/results/$MODEL_TYPE-sample
mkdir $MODEL_TYPE/$EXP_NAME
PYTHON_SCRIPT="query-auto-suggest/src/code/mpc/parallel_mpc_inference.py"
rm -f $MODEL_TYPE/$EXP_NAME/latency.tsv
start_time=$(gdate +%s%3N)
python3 "$PYTHON_SCRIPT" --input_file datasets/inputs/"$FILE"  --main_trie datasets/outputs/global-tries/main.mpc --suffix_trie datasets/outputs/global-tries/suffix.mpc --output_file datasets/results/global-tries-sample/completions_1000_sample_test.mpc --k_completions 100  --exp_name $EXP_NAME
python3 query-auto-suggest/src/code/mpc/00-reranker-scores.py --completions_path datasets/results/global-tries-sample  --op_path datasets/results/global-tries-sample --exp_type tries --context summary  --exp_name $EXP_NAME
python3 query-auto-suggest/src/code/mpc/01-reranker-algo.py --ip_path datasets/results/global-tries-sample --exp_type tries 
end_time=$(gdate +%s%3N)
execution_time=$((end_time - start_time))
echo "Execution time: $execution_time milliseconds"
echo -e "$i\t$MODEL_TYPE\t$FILE\t$execution_time" >> $MODEL_TYPE/$EXP_NAME/latency.tsv


# Title + URL + YAKE
MODEL_TYPE='global-tries'
EXP_NAME='title-url-yake'
echo "running $MODEL_TYPE - $EXP_NAME ......"
RUNS=1
mkdir $MODEL_TYPE
mkdir datasets/results/$MODEL_TYPE-sample
mkdir $MODEL_TYPE/$EXP_NAME
PYTHON_SCRIPT="query-auto-suggest/src/code/mpc/parallel_mpc_inference.py"
rm -f $MODEL_TYPE/$EXP_NAME/latency.tsv
start_time=$(gdate +%s%3N)
python3 "$PYTHON_SCRIPT" --input_file datasets/inputs/"$FILE"  --main_trie datasets/outputs/global-tries/main.mpc --suffix_trie datasets/outputs/global-tries/suffix.mpc --output_file datasets/results/global-tries-sample/completions_1000_sample_test.mpc --k_completions 100  --exp_name $EXP_NAME
python3 query-auto-suggest/src/code/mpc/00-reranker-scores.py --completions_path datasets/results/global-tries-sample  --op_path datasets/results/global-tries-sample --exp_type tries --context yake  --exp_name $EXP_NAME
python3 query-auto-suggest/src/code/mpc/01-reranker-algo.py --ip_path datasets/results/global-tries-sample --exp_type tries 
end_time=$(gdate +%s%3N)
execution_time=$((end_time - start_time))
echo "Execution time: $execution_time milliseconds"
echo -e "$i\t$MODEL_TYPE\t$FILE\t$execution_time" >> $MODEL_TYPE/$EXP_NAME/latency.tsv


#  Title + URL + RAG Sparse
MODEL_TYPE='global-tries'
EXP_NAME='title-url-ragsparse'
echo "running $MODEL_TYPE - $EXP_NAME ......"
RUNS=1
mkdir $MODEL_TYPE
mkdir datasets/results/$MODEL_TYPE-sample
mkdir $MODEL_TYPE/$EXP_NAME
PYTHON_SCRIPT="query-auto-suggest/src/code/mpc/parallel_mpc_inference.py"
rm -f $MODEL_TYPE/$EXP_NAME/latency.tsv
start_time=$(gdate +%s%3N)
python3 "$PYTHON_SCRIPT" --input_file datasets/inputs/"$FILE"  --main_trie datasets/outputs/global-tries/main.mpc --suffix_trie datasets/outputs/global-tries/suffix.mpc --output_file datasets/results/global-tries-sample/completions_1000_sample_test.mpc --k_completions 100  --exp_name $EXP_NAME
python3 query-auto-suggest/src/code/mpc/00-reranker-scores.py --completions_path datasets/results/global-tries-sample  --op_path datasets/results/global-tries-sample --exp_type tries --context rag_sparse_onedoc  --exp_name $EXP_NAME
python3 query-auto-suggest/src/code/mpc/01-reranker-algo.py --ip_path datasets/results/global-tries-sample --exp_type tries 
end_time=$(gdate +%s%3N)
execution_time=$((end_time - start_time))
echo "Execution time: $execution_time milliseconds"
echo -e "$i\t$MODEL_TYPE\t$FILE\t$execution_time" >> $MODEL_TYPE/$EXP_NAME/latency.tsv


#  Title + URL + RAG Dense
MODEL_TYPE='global-tries'
EXP_NAME='title-url-ragdense'
echo "running $MODEL_TYPE - $EXP_NAME ......"
RUNS=1
mkdir $MODEL_TYPE
mkdir datasets/results/$MODEL_TYPE-sample
mkdir $MODEL_TYPE/$EXP_NAME
PYTHON_SCRIPT="query-auto-suggest/src/code/mpc/parallel_mpc_inference.py"
rm -f $MODEL_TYPE/$EXP_NAME/latency.tsv
start_time=$(gdate +%s%3N)
python3 "$PYTHON_SCRIPT" --input_file datasets/inputs/"$FILE"  --main_trie datasets/outputs/global-tries/main.mpc --suffix_trie datasets/outputs/global-tries/suffix.mpc --output_file datasets/results/global-tries-sample/completions_1000_sample_test.mpc --k_completions 100  --exp_name $EXP_NAME
python3 query-auto-suggest/src/code/mpc/00-reranker-scores.py --completions_path datasets/results/global-tries-sample  --op_path datasets/results/global-tries-sample --exp_type tries --context rag_dense_onedoc  --exp_name $EXP_NAME
python3 query-auto-suggest/src/code/mpc/01-reranker-algo.py --ip_path datasets/results/global-tries-sample --exp_type tries 
end_time=$(gdate +%s%3N)
execution_time=$((end_time - start_time))
echo "Execution time: $execution_time milliseconds"
echo -e "$i\t$MODEL_TYPE\t$FILE\t$execution_time" >> $MODEL_TYPE/$EXP_NAME/latency.tsv




#  Title + URL + RAG Simdoc
MODEL_TYPE='global-tries'
EXP_NAME='title-url-ragsimdoc'
echo "running $MODEL_TYPE - $EXP_NAME ......"
RUNS=1
mkdir $MODEL_TYPE
mkdir datasets/results/$MODEL_TYPE-sample
mkdir $MODEL_TYPE/$EXP_NAME
PYTHON_SCRIPT="query-auto-suggest/src/code/mpc/parallel_mpc_inference.py"
rm -f $MODEL_TYPE/$EXP_NAME/latency.tsv
start_time=$(gdate +%s%3N)
python3 "$PYTHON_SCRIPT" --input_file datasets/inputs/"$FILE"  --main_trie datasets/outputs/global-tries/main.mpc --suffix_trie datasets/outputs/global-tries/suffix.mpc --output_file datasets/results/global-tries-sample/completions_1000_sample_test.mpc --k_completions 100  --exp_name $EXP_NAME
python3 query-auto-suggest/src/code/mpc/00-reranker-scores.py --completions_path datasets/results/global-tries-sample  --op_path datasets/results/global-tries-sample --exp_type tries --context rag_dense_simdoc  --exp_name $EXP_NAME
python3 query-auto-suggest/src/code/mpc/01-reranker-algo.py --ip_path datasets/results/global-tries-sample --exp_type tries 
end_time=$(gdate +%s%3N)
execution_time=$((end_time - start_time))
echo "Execution time: $execution_time milliseconds"
echo -e "$i\t$MODEL_TYPE\t$FILE\t$execution_time" >> $MODEL_TYPE/$EXP_NAME/latency.tsv

