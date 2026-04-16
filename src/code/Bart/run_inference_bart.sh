#!/bin/bash

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export CHECKPOINT_DIR='checkpoints_bart_no_doc'  #'checkpoints_bart_yake', #checkpoints_bart_no_doc, 'checkpoints_bart-rag-dense-onedoc' #'checkpoints_bart_title_url_doc'  checkpoints_bart-rag-sparse-onedoc, 
export EXP_TYPE='no_doc' #'url_doc' ,'no_doc','rag_sparse_onedoc','summary','rag_dense_onedoc','rag_dense_simdoc', 'url','yake'

# Url
# export CHECKPOINT_DIR='checkpoints_bart_url'  #'checkpoints_bart_yake', #checkpoints_bart_no_doc, 'checkpoints_bart-rag-dense-onedoc' #'checkpoints_bart_title_url_doc'  checkpoints_bart-rag-sparse-onedoc, 
# export EXP_TYPE='url'


# # summary
# export CHECKPOINT_DIR='checkpoints_bart_summary_300' 
# export EXP_TYPE='summary'

# # yake 
#export CHECKPOINT_DIR='checkpoints_bart_yake' 
#export EXP_TYPE='yake'


# # rag-dense-onedoc
# export CHECKPOINT_DIR='checkpoints_bart-rag-dense-onedoc' 
# export EXP_TYPE='rag_dense_onedoc'

#  #rag-sparse-onedoc
#export CHECKPOINT_DIR='checkpoints_bart-rag-sparse-onedoc' 
#export EXP_TYPE='rag_sparse_onedoc'

# # rag-dense-simdoc
# export CHECKPOINT_DIR='checkpoints_bart-rag-dense-simdoc' 
# export EXP_TYPE='rag_dense_simdoc'


export BATCH_SIZE=2
export MODEL_NAME='facebook/bart-base'

# Create results directories
mkdir -p datasets/results/$MODEL_NAME/$EXP_TYPE

# Run inference on all splits
echo "seen_query-unseen_doc"
python src/code/Bart/infer.py \
  --inp datasets/master/queries-inference/test_formatted_seen_query-unseen_doc_test.tsv \
  --out datasets/results/$MODEL_NAME/$EXP_TYPE/completions_seen_query-unseen_doc_test.mpc \
  --doc datasets/master/docs/trec_test.csv \
  --ckpt $CHECKPOINT_DIR \
  --mdmax_length 48 \
  --input_type $EXP_TYPE \
  --model_name $MODEL_NAME \
  --beam_size 25 \
  --bs $BATCH_SIZE


# echo "unseen_query-unseen_doc"
# python src/code/Bart/infer.py \
#   --inp datasets/master/queries-inference/test_formatted_unseen_query-unseen_doc_test.tsv \
#   --out datasets/results/$MODEL_NAME/$EXP_TYPE/completions_unseen_query-unseen_doc_test.mpc \
#   --doc datasets/master/docs/trec_test.csv \
#   --ckpt $CHECKPOINT_DIR \
#   --mdmax_length 48 \
#   --input_type $EXP_TYPE \
#   --model_name $MODEL_NAME \
#   --beam_size 25 \
#   --bs $BATCH_SIZE


# echo "seen_query-seen_doc"
# python src/code/Bart/infer.py \
#   --inp datasets/master/queries-inference/test_formatted_seen_query-seen_doc_test.tsv \
#   --out datasets/results/$MODEL_NAME/$EXP_TYPE/completions_seen_query-seen_doc_test.mpc \
#   --doc datasets/master/docs/trec_test.csv \
#   --ckpt $CHECKPOINT_DIR \
#   --mdmax_length 48 \
#   --input_type $EXP_TYPE \
#   --model_name $MODEL_NAME \
#   --beam_size 25 \
#   --bs $BATCH_SIZE

echo "unseen_query-seen_doc"
python src/code/Bart/infer.py \
  --inp datasets/master/queries-inference/test_formatted_unseen_query-seen_doc_test.tsv \
  --out datasets/results/$MODEL_NAME/$EXP_TYPE/completions_unseen_query-seen_doc_test.mpc \
  --doc datasets/master/docs/trec_test.csv \
  --ckpt $CHECKPOINT_DIR \
  --mdmax_length 48 \
  --input_type $EXP_TYPE \
  --model_name $MODEL_NAME \
  --beam_size 25 \
  --bs $BATCH_SIZE
