#!/bin/bash

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
# export CHECKPOINT_DIR='checkpoints_bart_no_doc'  #'checkpoints_bart_yake', #checkpoints_bart_no_doc, 'checkpoints_bart-rag-dense-onedoc' #'checkpoints_bart_title_url_doc'  checkpoints_bart-rag-sparse-onedoc, 
# export EXP_TYPE='no_doc' #'url_doc' ,'no_doc','rag_sparse_onedoc','summary','rag_dense_onedoc','rag_dense_simdoc', 'url','yake'

# url doc
# export CHECKPOINT_DIR='checkpoints_bart_url_doc'
# export EXP_TYPE='url_doc'
# Url
# export CHECKPOINT_DIR='checkpoints_bart_url'  #'checkpoints_bart_yake', #checkpoints_bart_no_doc, 'checkpoints_bart-rag-dense-onedoc' #'checkpoints_bart_title_url_doc'  checkpoints_bart-rag-sparse-onedoc, 
# export EXP_TYPE='url'


# # summary
#export CHECKPOINT_DIR='checkpoints_bart_summary_300' 
#export EXP_TYPE='summary'

# # yake 
#export CHECKPOINT_DIR='checkpoints_bart_yake' 
#export EXP_TYPE='yake'


# # rag-dense-onedoc
# export CHECKPOINT_DIR='checkpoints_bart-rag-dense-onedoc' 
# export EXP_TYPE='rag_dense_onedoc'

#  #rag-sparse-onedoc
# export CHECKPOINT_DIR='checkpoints_bart-rag-sparse-onedoc' 
# export EXP_TYPE='rag_sparse_onedoc'

# # rag-dense-simdoc
export CHECKPOINT_DIR='checkpoints_bart-rag-dense-simdoc' 
export EXP_TYPE='rag_dense_simdoc'

export BATCH_SIZE=4 # title_url_doc 28 
export MODEL_NAME='facebook/bart-base'
# Run inference on all splits
export ALPHA=0.1
export BETA=0.2
export BIAS=40


# Create results directories
mkdir -p datasets/results/$MODEL_NAME/constrained/$EXP_TYPE
mkdir -p datasets/results/$MODEL_NAME/constrained/$EXP_TYPE/$ALPHA-$BETA-$BIAS 



export TRIE_PATH='datasets/master/bart/constrained/seq_list_doc_bart.pkl'

echo "seen_query-seen_doc"
python src/code/Bart/infer.py \
  --inp datasets/master/queries-inference/test_formatted_seen_query-seen_doc_test.tsv \
  --out datasets/results/$MODEL_NAME/constrained/$EXP_TYPE/$ALPHA-$BETA-$BIAS/completions_seen_query-seen_doc_test.mpc \
  --doc datasets/master/docs/trec_test.csv \
  --ckpt $CHECKPOINT_DIR \
  --mdmax_length 48 \
  --input_type $EXP_TYPE \
  --model_name $MODEL_NAME \
  --beam_size 25 \
  --bs $BATCH_SIZE \
  --trie_path $TRIE_PATH --alpha $ALPHA \
  --beta $BETA  \
  --bias_strength $BIAS \
  --use_trie


export ALPHA=0.5
export BETA=0.2
export BIAS=20

mkdir -p datasets/results/$MODEL_NAME/constrained/$EXP_TYPE
mkdir -p datasets/results/$MODEL_NAME/constrained/$EXP_TYPE/$ALPHA-$BETA-$BIAS 

echo "unseen_query-seen_doc"
python src/code/Bart/infer.py \
  --inp datasets/master/queries-inference/test_formatted_unseen_query-seen_doc_test.tsv \
  --out datasets/results/$MODEL_NAME/constrained/$EXP_TYPE/$ALPHA-$BETA-$BIAS/completions_unseen_query-seen_doc_test.mpc \
  --doc datasets/master/docs/trec_test.csv \
  --ckpt $CHECKPOINT_DIR \
  --mdmax_length 48 \
  --input_type $EXP_TYPE \
  --model_name $MODEL_NAME \
  --beam_size 25 \
  --bs $BATCH_SIZE \
  --trie_path $TRIE_PATH --alpha $ALPHA \
  --beta $BETA  \
  --bias_strength $BIAS \
  --use_trie

######### SAMPLE TEST
# export CUDA_VISIBLE_DEVICES=0
# export CHECKPOINT_DIR='checkpoints_bart_no_doc'  #'checkpoints_bart_yake', #checkpoints_bart_no_doc, 'checkpoints_bart-rag-dense-onedoc' #'checkpoints_bart_title_url_doc'  checkpoints_bart-rag-sparse-onedoc, 
# export EXP_TYPE='no_doc' #'url_doc' ,'no_doc','rag_sparse_onedoc','summary','rag_dense_onedoc','rag_dense_simdoc', 'url','yake'
# export BATCH_SIZE=2 # title_url_doc 28 
# export MODEL_NAME='facebook/bart-base'

# export ALPHA=0.1
# export BETA=0.1
# export BIAS=100

# mkdir -p datasets/results/$MODEL_NAME/constrained/$EXP_TYPE/
# mkdir -p datasets/results/$MODEL_NAME/constrained/$EXP_TYPE/$ALPHA-$BETA-$BIAS 


# echo "seen_query-seen_doc"
# python src/code/Bart/infer.py \
#   --inp datasets/master/queries-inference/test_formatted_seen_query-seen_doc_test_10.tsv \
#   --out datasets/results/$MODEL_NAME/constrained/$EXP_TYPE/$ALPHA-$BETA-$BIAS/completions_seen_query-seen_doc_test.mpc \
#   --doc datasets/master/docs/trec_test.csv \
#   --ckpt $CHECKPOINT_DIR \
#   --mdmax_length 48 \
#   --input_type $EXP_TYPE \
#   --model_name $MODEL_NAME \
#   --beam_size 25 \
#   --bs $BATCH_SIZE \
#   --trie_path $TRIE_PATH --alpha $ALPHA \
#   --beta $BETA  \
#   --bias_strength $BIAS \
#   --use_trie


# head -5 datasets/results/facebook/bart-base/constrained/no_doc/0.1-0.1-100/completions_seen_query-seen_doc_test.mpc
