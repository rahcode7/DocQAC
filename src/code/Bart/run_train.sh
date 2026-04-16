##----------------------------------------------------------------------------------------- No Doc 

export MODEL_NAME='facebook/bart-base'
export CHECKPOINT_DIR='checkpoints_bart_no_doc'
export CUDA_VISIBLE_DEVICES=0  # Set your gpu ids
export NUM_GPUS=1
export EPOCHS=30 # 
export BATCH_SIZE=18 
export EXP_TYPE='no_doc' # url,url_doc,yake,full_doc,summary # full_doc is prefix only

rm -rf $CHECKPOINT_DIR
mkdir $CHECKPOINT_DIR

accelerate launch --multi_gpu --num_processes $NUM_GPUS src/code/Bart/train.py --model_dir $CHECKPOINT_DIR --train_data datasets/master/queries/train.csv --train_doc datasets/master/docs/trec_train.csv --val_data datasets/master/queries/val.csv --val_doc datasets/master/docs/trec_val.csv --bs $BATCH_SIZE --input_type $EXP_TYPE --wandb --num_epochs $EPOCHS --eval_every 10000000000 --model_name $MODEL_NAME

##----------------------------------------------------------------------------------------- Url doc 

export MODEL_NAME='facebook/bart-base'
export CHECKPOINT_DIR='checkpoints_bart_url_doc'
export CUDA_VISIBLE_DEVICES=2  # Set your gpu ids
export NUM_GPUS=1
export EPOCHS=30 # 
export BATCH_SIZE=8
export EXP_TYPE='url_doc' # url,url_doc,yake,full_doc,summary # full_doc is prefix only

rm -rf $CHECKPOINT_DIR
mkdir $CHECKPOINT_DIR

accelerate launch --num_processes $NUM_GPUS src/code/Bart/train.py --model_dir $CHECKPOINT_DIR --train_data datasets/master/queries/train.csv --train_doc datasets/master/docs/trec_train.csv --val_data datasets/master/queries/val.csv --val_doc datasets/master/docs/trec_val.csv --bs $BATCH_SIZE --input_type $EXP_TYPE --wandb --num_epochs $EPOCHS --eval_every 10000000000 --model_name $MODEL_NAME

##----------------------------------------------------------------------------------------- Url 

export MODEL_NAME='facebook/bart-base'
export CHECKPOINT_DIR='checkpoints_bart_url'
export CUDA_VISIBLE_DEVICES=0  # Set your gpu ids
export NUM_GPUS=1
export EPOCHS=30 # 
export BATCH_SIZE=18 
export EXP_TYPE='url' # url,url_doc,yake,full_doc,summary # full_doc is prefix only

rm -rf $CHECKPOINT_DIR
mkdir $CHECKPOINT_DIR

accelerate launch  --num_processes $NUM_GPUS src/code/Bart/train.py --model_dir $CHECKPOINT_DIR --train_data datasets/master/queries/train.csv --train_doc datasets/master/docs/trec_train.csv --val_data datasets/master/queries/val.csv --val_doc datasets/master/docs/trec_val.csv --bs $BATCH_SIZE --input_type $EXP_TYPE --wandb --num_epochs $EPOCHS --eval_every 10000000000 --model_name $MODEL_NAME


##----------------------------------------------------------------------------------------- Sum

export MODEL_NAME='facebook/bart-base'
export CHECKPOINT_DIR='checkpoints_bart_summary_300'
export CUDA_VISIBLE_DEVICES=1 # Set your gpu ids
export NUM_GPUS=1
export EPOCHS=30 # 
export BATCH_SIZE=8 
export EXP_TYPE='summary' 

rm -rf $CHECKPOINT_DIR
mkdir $CHECKPOINT_DIR

accelerate launch  --num_processes $NUM_GPUS src/code/Bart/train.py --model_dir $CHECKPOINT_DIR --train_data datasets/master/queries/train.csv --train_doc datasets/master/docs/trec_train.csv --val_data datasets/master/queries/val.csv --val_doc datasets/master/docs/trec_val.csv --bs $BATCH_SIZE --input_type $EXP_TYPE --wandb --num_epochs $EPOCHS --eval_every 10000000000 --model_name $MODEL_NAME


##-----------------------------------------------------------------------------------------Yake 

export MODEL_NAME='facebook/bart-base'
export CHECKPOINT_DIR='checkpoints_bart_yake'
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Set your gpu ids
export NUM_GPUS=4
export EPOCHS=30 # 
export BATCH_SIZE=8 
export EXP_TYPE='yake' # url,url_doc,yake,full_doc,summary # full_doc is prefix only

rm -rf $CHECKPOINT_DIR
mkdir $CHECKPOINT_DIR

accelerate launch --multi_gpu --num_processes $NUM_GPUS src/code/Bart/train.py --model_dir $CHECKPOINT_DIR --train_data datasets/master/queries/train.csv --train_doc datasets/master/docs/trec_train.csv --val_data datasets/master/queries/val.csv --val_doc datasets/master/docs/trec_val.csv --bs $BATCH_SIZE --input_type $EXP_TYPE --wandb --num_epochs $EPOCHS --eval_every 10000000000 --model_name $MODEL_NAME

##-----------------------------------------------------------------------------------------Rag dense

export MODEL_NAME='facebook/bart-base'
export CHECKPOINT_DIR='checkpoints_bart-rag-dense-onedoc'
export CUDA_VISIBLE_DEVICES=0  # Set your gpu ids
export NUM_GPUS=1
export EPOCHS=30 # 
export BATCH_SIZE=8 
export EXP_TYPE='rag_dense_onedoc' 

rm -rf $CHECKPOINT_DIR
mkdir $CHECKPOINT_DIR

accelerate launch --num_processes $NUM_GPUS src/code/Bart/train.py --model_dir $CHECKPOINT_DIR --train_data datasets/master/queries/train.csv --train_doc datasets/master/docs/trec_train.csv --val_data datasets/master/queries/val.csv --val_doc datasets/master/docs/trec_val.csv --bs $BATCH_SIZE --input_type $EXP_TYPE --wandb --num_epochs $EPOCHS --eval_every 10000000000 --model_name $MODEL_NAME

##-----------------------------------------------------------------------------------------Rag sparse

export MODEL_NAME='facebook/bart-base'
export CHECKPOINT_DIR='checkpoints_bart-rag-sparse-onedoc'
export CUDA_VISIBLE_DEVICES=1  # Set your gpu ids
export NUM_GPUS=1
export EPOCHS=30 # 
export BATCH_SIZE=8 
export EXP_TYPE='rag_sparse_onedoc' 

rm -rf $CHECKPOINT_DIR
mkdir $CHECKPOINT_DIR

accelerate launch --num_processes $NUM_GPUS src/code/Bart/train.py --model_dir $CHECKPOINT_DIR --train_data datasets/master/queries/train.csv --train_doc datasets/master/docs/trec_train.csv --val_data datasets/master/queries/val.csv --val_doc datasets/master/docs/trec_val.csv --bs $BATCH_SIZE --input_type $EXP_TYPE --wandb --num_epochs $EPOCHS --eval_every 10000000000 --model_name $MODEL_NAME

##-----------------------------------------------------------------------------------------Rag simdoc 

export MODEL_NAME='facebook/bart-base'
export CHECKPOINT_DIR='checkpoints_bart-rag-dense-simdoc'
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Set your gpu ids
export NUM_GPUS=4
export EPOCHS=30 # 
export BATCH_SIZE=8 
export EXP_TYPE='rag_dense_simdoc' 

rm -rf $CHECKPOINT_DIR
mkdir $CHECKPOINT_DIR

accelerate launch --multi_gpu --num_processes $NUM_GPUS src/code/Bart/train.py --model_dir $CHECKPOINT_DIR --train_data datasets/master/queries/train.csv --train_doc datasets/master/docs/trec_train.csv --val_data datasets/master/queries/val.csv --val_doc datasets/master/docs/trec_val.csv --bs $BATCH_SIZE --input_type $EXP_TYPE --wandb --num_epochs $EPOCHS --eval_every 10000000000 --model_name $MODEL_NAME


