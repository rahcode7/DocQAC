`

#### Arguments
```
parser.add_argument('--train_data', type=str, required=True)
parser.add_argument('--train_doc', type=str, required=True)
parser.add_argument('--val_doc', type=str, required=True)
parser.add_argument('--model_dir', type=str)
parser.add_argument('--num_epochs', type=int, default=40)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--val_data', type=str, default=None)
parser.add_argument('--bs',  type=int, default=4)
parser.add_argument('--ckpt', type=str, default=None)
parser.add_argument('--tkmax_length', type=int, default=512)
parser.add_argument('--mdmax_length', type=int, default=512)
parser.add_argument('--initial_eval', action='store_true')
parser.add_argument('--eval_every', type=int, default=7500)
parser.add_argument('--wandb', action='store_true')
parser.add_argument('--dev', action='store_true')
parser.add_argument('--model_name', type=str, default="t5-base")
parser.add_argument('--input_type', type=str, default="full_doc")
parser.add_argument('--suffix_trie_path', type=str, default="datasets/outputs/global-tries/suffix.mpc")
parser.add_argument('--main_trie_path', type=str, default="datasets/outputs/global-tries/main.mpc")
parser.add_argument('--k_comp', type=int, default=10)


```

####  Step0. Requirements
```
conda create -n autosuggest python=3.10.15
conda activate autosuggest
pip install -r src/code/T5/requirements.txt
```

### Experiment 1 RAG - Title,Url and Sparse 1 Doc + T5 
##### Step 2. Training 

```
export MODEL_NAME='t5-small'
export CHECKPOINT_DIR='checkpoints_t5-rag-sparse-onedoc'
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Set your gpu ids
export NUM_GPUS=4
export EPOCHS=40 # past experiments 
export BATCH_SIZE=36 # 36 for 64 GB  # 28 for 48GB - training took 40 epochs 2 days with docs
export EXP_TYPE='rag_sparse_onedoc'


rm -rf $CHECKPOINT_DIR
mkdir $CHECKPOINT_DIR

wandb login --relogin


accelerate launch --multi_gpu --num_processes $NUM_GPUS src/code/t5/train.py  --model_dir $CHECKPOINT_DIR --train_data  datasets/master/queries/train.csv --train_doc datasets/master/docs/trec_train.csv --val_data datasets/master/queries/val.csv --val_doc datasets/master/docs/trec_val.csv --bs $BATCH_SIZE   --input_type $EXP_TYPE  --num_epochs $EPOCHS --eval_every 10000000000 --model_name $MODEL_NAME --wandb
```

##### Step3. Inference 
```
export CUDA_VISIBLE_DEVICES=0
export CHECKPOINT_DIR='checkpoints_t5-rag-sparse-onedoc'
export MODEL_NAME='t5-small'
export EXP_TYPE='rag_sparse_onedoc'
export BATCH_SIZE=36


mkdir datasets/results
mkdir datasets/results/$MODEL_NAME
mkdir datasets/results/$MODEL_NAME/$EXP_TYPE

python src/code/T5/infer.py 
        --inp datasets/master/queries-inference/test_formatted_seen_query-unseen_doc_test.tsv
        --out datasets/results/$MODEL_NAME/$EXP_TYPE/completions_seen_query-unseen_doc_test.mpc 
        --doc datasets/master/docs/trec_test.csv
        --ckpt $CHECKPOINT_DIR
        --mdmax_length 48 
        --input_type $EXP_TYPE 
        --model_name $MODEL_NAME
        --beam_size 25 
        --bs $BATCH_SIZE

export CUDA_VISIBLE_DEVICES=1
python src/code/T5/infer.py 
        --inp datasets/master/queries-inference/test_formatted_unseen_query-unseen_doc_test.tsv
        --out datasets/results/$MODEL_NAME/$EXP_TYPE/completions_unseen_query-unseen_doc_test.mpc
        --doc datasets/master/docs/trec_test.csv
        --ckpt $CHECKPOINT_DIR
        --mdmax_length 48 
        --input_type $EXP_TYPE 
        --model_name $MODEL_NAME
        --beam_size 25 
        --bs $BATCH_SIZE

export CUDA_VISIBLE_DEVICES=2
python src/code/T5/infer.py  
        --inp datasets/master/queries-inference/test_formatted_seen_query-seen_doc_test.tsv
        --out datasets/results/$MODEL_NAME/$EXP_TYPE/completions_seen_query-seen_doc_test.mpc
        --doc datasets/master/docs/trec_test.csv
        --ckpt $CHECKPOINT_DIR
        --mdmax_length 48 
        --input_type $EXP_TYPE 
        --model_name $MODEL_NAME
        --beam_size 25 
        --bs $BATCH_SIZE

export CUDA_VISIBLE_DEVICES=3
python src/code/T5/infer.py 
        --inp datasets/master/queries-inference/test_formatted_unseen_query-seen_doc_test.tsv
        --out datasets/results/$MODEL_NAME/$EXP_TYPE/completions_unseen_query-seen_doc_test.mpc
        --doc datasets/master/docs/trec_test.csv
        --ckpt $CHECKPOINT_DIR
        --mdmax_length 48 
        --input_type $EXP_TYPE 
        --model_name $MODEL_NAME
        --beam_size 25 
        --bs $BATCH_SIZE
```



### Experiment 2: RAG - Title,Url and Dense 1 Doc + T5
##### Download vector stores
```
gdown https://drive.google.com/uc\?id\=1-BkD-OavrvfwCohv-4HoULjyhkOX7Av1
mkdir datasets/rag datasets/rag/vector_stores
unzip rag-all.zip  -d  datasets/rag/vector_stores
```

##### Step2. Training
```
export MODEL_NAME='t5-small'
export CHECKPOINT_DIR='checkpoints_t5-rag-dense-onedoc'
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Set your gpu ids
export NUM_GPUS=4
mkdir datasets/results/$MODEL_NAME/$EXP_TYPE

export EPOCHS=40 # past experiments 
export BATCH_SIZE=36 # 36 for 64 GB  # 28 for 48GB - training took 40 epochs 2 days with docs

rm -rf $CHECKPOINT_DIR
mkdir $CHECKPOINT_DIR

wandb login --relogin

export EXP_TYPE='rag_dense_onedoc'

accelerate launch --multi_gpu --num_processes $NUM_GPUS src/code/t5/train.py  --model_dir $CHECKPOINT_DIR --train_data  datasets/master/queries/train.csv --train_doc datasets/master/docs/trec_train.csv --val_data datasets/master/queries/val.csv --val_doc datasets/master/docs/trec_val.csv --bs $BATCH_SIZE   --input_type $EXP_TYPE  --num_epochs $EPOCHS --eval_every 10000000000 --model_name $MODEL_NAME --wandb
```


###### Step3. Inference 
```
export CUDA_VISIBLE_DEVICES=0
export CHECKPOINT_DIR='checkpoints_t5-rag-dense-onedoc'
export MODEL_NAME='t5-small'
export EXP_TYPE='rag_dense_onedoc'
export BATCH_SIZE=36


mkdir datasets/results
mkdir datasets/results/$MODEL_NAME
mkdir datasets/results/$MODEL_NAME/$EXP_TYPE

python src/code/T5/infer.py 
        --inp datasets/master/queries-inference/test_formatted_seen_query-unseen_doc_test.tsv
        --out datasets/results/$MODEL_NAME/$EXP_TYPE/completions_seen_query-unseen_doc_test.mpc 
        --doc datasets/master/docs/trec_test.csv
        --ckpt $CHECKPOINT_DIR
        --mdmax_length 48 
        --input_type $EXP_TYPE 
        --model_name $MODEL_NAME
        --beam_size 25 
        --bs $BATCH_SIZE

export CUDA_VISIBLE_DEVICES=1
python src/code/T5/infer.py 
        --inp datasets/master/queries-inference/test_formatted_unseen_query-unseen_doc_test.tsv
        --out datasets/results/$MODEL_NAME/$EXP_TYPE/completions_unseen_query-unseen_doc_test.mpc
        --doc datasets/master/docs/trec_test.csv
        --ckpt $CHECKPOINT_DIR
        --mdmax_length 48 
        --input_type $EXP_TYPE 
        --model_name $MODEL_NAME
        --beam_size 25 
        --bs $BATCH_SIZE

export CUDA_VISIBLE_DEVICES=2
python src/code/T5/infer.py  
        --inp datasets/master/queries-inference/test_formatted_seen_query-seen_doc_test.tsv
        --out datasets/results/$MODEL_NAME/$EXP_TYPE/completions_seen_query-seen_doc_test.mpc
        --doc datasets/master/docs/trec_test.csv
        --ckpt $CHECKPOINT_DIR
        --mdmax_length 48 
        --input_type $EXP_TYPE 
        --model_name $MODEL_NAME
        --beam_size 25 
        --bs $BATCH_SIZE

export CUDA_VISIBLE_DEVICES=3
python src/code/T5/infer.py 
        --inp datasets/master/queries-inference/test_formatted_unseen_query-seen_doc_test.tsv
        --out datasets/results/$MODEL_NAME/$EXP_TYPE/completions_unseen_query-seen_doc_test.mpc
        --doc datasets/master/docs/trec_test.csv
        --ckpt $CHECKPOINT_DIR
        --mdmax_length 48 
        --input_type $EXP_TYPE 
        --model_name $MODEL_NAME
        --beam_size 25 
        --bs $BATCH_SIZE
```


### Experiment 3 Document Summarization and Title +  T5 

##### Step 2. Training 
```
export MODEL_NAME='t5-small'
export CHECKPOINT_DIR='checkpoints_t5_summary_300'
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Set your gpu ids
export NUM_GPUS=4

export EPOCHS=40 
export BATCH_SIZE=2 # 36 for 64 GB  # 28 for 48GB - training took 40 epochs 2 days with docs

rm -rf $CHECKPOINT_DIR
mkdir $CHECKPOINT_DIR

wandb login --relogin
export EXP_TYPE='summary'

accelerate launch --multi_gpu --num_processes $NUM_GPUS src/code/T5/train.py  --model_dir $CHECKPOINT_DIR --train_data  datasets/master/queries/train.csv --train_doc datasets/master/summaries/trec_train_summary_300words.csv --val_data datasets/master/queries/val.csv --val_doc datasets/master/summaries/trec_val_summary_300words.csv --bs $BATCH_SIZE   --input_type $EXP_TYPE --wandb  --num_epochs $EPOCHS --eval_every 10000000000 --model_name $MODEL_NAME --ckpt $CHECKPOINT_DIR/epoch_28.pth
```

##### Step3. Inference 
```
export CUDA_VISIBLE_DEVICES=0
export MODEL_NAME='t5-small'
export CHECKPOINT_DIR='checkpoints_t5_summary_300'
export EXP_TYPE='summary' 
export BATCH_SIZE=36


mkdir datasets/results
mkdir datasets/results/$MODEL_NAME
mkdir datasets/results/$MODEL_NAME/$EXP_TYPE

python src/code/T5/infer.py
        --inp datasets/master/queries-inference/test_formatted_seen_query-unseen_doc_test.tsv
        --out datasets/results/$MODEL_NAME/$EXP_TYPE/completions_seen_query-unseen_doc_test.mpc 
        --doc datasets/master/summaries/trec_test_summary_300words.csv
        --ckpt $CHECKPOINT_DIR
        --mdmax_length 48 
        --input_type $EXP_TYPE 
        --model_name $MODEL_NAME
        --beam_size 25 
        --bs $BATCH_SIZE

export CUDA_VISIBLE_DEVICES=1
python src/code/T5/infer.py 
        --inp datasets/master/queries-inference/test_formatted_unseen_query-unseen_doc_test.tsv
        --out datasets/results/$MODEL_NAME/$EXP_TYPE/completions_unseen_query-unseen_doc_test.mpc 
        --doc datasets/master/summaries/trec_test_summary_300words.csv
        --ckpt $CHECKPOINT_DIR
        --mdmax_length 48 
        --input_type $EXP_TYPE 
        --model_name $MODEL_NAME
        --beam_size 25 
        --bs $BATCH_SIZE

export CUDA_VISIBLE_DEVICES=2
python src/code/T5/infer.py
        --inp datasets/master/queries-inference/test_formatted_seen_query-seen_doc_test.tsv
        --out datasets/results/$MODEL_NAME/$EXP_TYPE/completions_seen_query-seen_doc_test.mpc
        --doc datasets/master/summaries/trec_test_summary_300words.csv 
        --ckpt $CHECKPOINT_DIR
        --mdmax_length 48 
        --input_type $EXP_TYPE 
        --model_name $MODEL_NAME
        --beam_size 25 
        --bs $BATCH_SIZE

export CUDA_VISIBLE_DEVICES=3
python src/code/T5/infer.py
        --inp datasets/master/queries-inference/test_formatted_unseen_query-seen_doc_test.tsv
        --out datasets/results/$MODEL_NAME/$EXP_TYPE/completions_unseen_query-seen_doc_test.mpc 
        --doc datasets/master/summaries/trec_test_summary_300words.csv
        --ckpt $CHECKPOINT_DIR
        --mdmax_length 48 
        --input_type $EXP_TYPE 
        --model_name $MODEL_NAME
        --beam_size 25 
        --bs $BATCH_SIZE
```

### Experiment 4  Title,Url and Document + T5 

##### Step 2. Training 
```
export MODEL_NAME='t5-small'
export CHECKPOINT_DIR='checkpoints_t5_title_url_doc'
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Set your gpu ids
export NUM_GPUS=4
export EPOCHS=40 # past experiments T5-
export BATCH_SIZE=36 # 36 for 64 GB  # 40 for 48GB 
export EXP_TYPE='url_doc'

rm -rf $CHECKPOINT
mkdir $CHECKPOINT

wandb login --relogin


accelerate launch --multi_gpu --num_processes $NUM_GPUS src/code/T5/train.py --model_dir $CHECKPOINT_DIR --train_data datasets/master/queries/train.csv --train_doc datasets/master/docs/trec_train.csv --val_data datasets/master/queries/val.csv --val_doc datasets/master/docs/trec_val.csv --bs $BATCH_SIZE --input_type $EXP_TYPE --wandb --num_epochs $EPOCHS --eval_every 10000000000 --model_name $MODEL_NAME --ckpt $CHECKPOINT_DIR/epoch_29.pth
```

##### Step3. Inference 
```
export CUDA_VISIBLE_DEVICES=0
export CHECKPOINT_DIR='checkpoints_t5_title_url_doc'
export MODEL_NAME='t5-small'
export EXP_TYPE='url_doc' 
export BATCH_SIZE=36

mkdir datasets/results
mkdir datasets/results/$MODEL_NAME
mkdir datasets/results/$MODEL_NAME/$EXP_TYPE

python src/code/T5/infer.py 
        --inp datasets/master/queries-inference/test_formatted_seen_query-unseen_doc_test.tsv
        --out datasets/results/$MODEL_NAME/$EXP_TYPE/completions_seen_query-unseen_doc_test.mpc 
        --doc datasets/master/docs/trec_test.csv
        --ckpt $CHECKPOINT_DIR
        --mdmax_length 48 
        --input_type $EXP_TYPE 
        --model_name $MODEL_NAME
        --beam_size 25 
        --bs $BATCH_SIZE

export CUDA_VISIBLE_DEVICES=1
python src/code/T5/infer.py 
        --inp datasets/master/queries-inference/test_formatted_unseen_query-unseen_doc_test.tsv
        --out datasets/results/$MODEL_NAME/$EXP_TYPE/completions_unseen_query-unseen_doc_test.mpc
        --doc datasets/master/docs/trec_test.csv
        --ckpt $CHECKPOINT_DIR
        --mdmax_length 48 
        --input_type $EXP_TYPE 
        --model_name $MODEL_NAME
        --beam_size 25 
        --bs $BATCH_SIZE

export CUDA_VISIBLE_DEVICES=2
python src/code/T5/infer.py  
        --inp datasets/master/queries-inference/test_formatted_seen_query-seen_doc_test.tsv
        --out datasets/results/$MODEL_NAME/$EXP_TYPE/completions_seen_query-seen_doc_test.mpc
        --doc datasets/master/docs/trec_test.csv
        --ckpt $CHECKPOINT_DIR
        --mdmax_length 48 
        --input_type $EXP_TYPE 
        --model_name $MODEL_NAME
        --beam_size 25 
        --bs $BATCH_SIZE

export CUDA_VISIBLE_DEVICES=3
python src/code/T5/infer.py 
        --inp datasets/master/queries-inference/test_formatted_unseen_query-seen_doc_test.tsv
        --out datasets/results/$MODEL_NAME/$EXP_TYPE/completions_unseen_query-seen_doc_test.mpc
        --doc datasets/master/docs/trec_test.csv
        --ckpt $CHECKPOINT_DIR
        --mdmax_length 48 
        --input_type $EXP_TYPE 
        --model_name $MODEL_NAME
        --beam_size 25 
        --bs $BATCH_SIZE
```

### Experiment 5  Title,Url + T5 

##### Step 2. Training 
```
export MODEL_NAME='t5-small'
export CHECKPOINT_DIR='checkpoints_t5_title_url'
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Set your gpu ids
export EPOCHS=40 
export BATCH_SIZE=36 # 36 for 64 GB  # 40 for 48GB 
export NUM_GPUS=4
export EXP_TYPE='url' 

rm -rf $CHECKPOINT
mkdir $CHECKPOINT

wandb login --relogin

accelerate launch --multi_gpu --num_processes $NUM_GPUS src/code/T5/train.py --model_dir $CHECKPOINT_DIR --train_data datasets/master/queries/train.csv --train_doc datasets/master/docs/trec_train.csv --val_data datasets/master/queries/val.csv --val_doc datasets/master/docs/trec_val.csv --bs $BATCH_SIZE --input_type $EXP_TYPE --wandb --num_epochs $EPOCHS --eval_every 10000000000 --model_name $MODEL_NAME
```

##### Step3. Inference 
```
export CUDA_VISIBLE_DEVICES=0
export CHECKPOINT_DIR='checkpoints_t5_title_url'
export EXP_TYPE='url' 
export BATCH_SIZE=36
export MODEL_NAME='t5-small'


mkdir datasets/results
mkdir datasets/results/$MODEL_NAME
mkdir datasets/results/$MODEL_NAME/$EXP_TYPE


python src/code/T5/infer.py 
        --inp datasets/master/queries-inference/test_formatted_seen_query-unseen_doc_test.tsv
        --out datasets/results/$MODEL_NAME/$EXP_TYPE/completions_seen_query-unseen_doc_test.mpc 
        --doc datasets/master/docs/trec_test.csv
        --ckpt $CHECKPOINT_DIR
        --mdmax_length 48 
        --input_type $EXP_TYPE 
        --model_name $MODEL_NAME
        --beam_size 25 
        --bs $BATCH_SIZE

export CUDA_VISIBLE_DEVICES=1
python src/code/T5/infer.py 
        --inp datasets/master/queries-inference/test_formatted_unseen_query-unseen_doc_test.tsv
        --out datasets/results/$MODEL_NAME/$EXP_TYPE/completions_unseen_query-unseen_doc_test.mpc
        --doc datasets/master/docs/trec_test.csv
        --ckpt $CHECKPOINT_DIR
        --mdmax_length 48 
        --input_type $EXP_TYPE 
        --model_name $MODEL_NAME
        --beam_size 25 
        --bs $BATCH_SIZE

export CUDA_VISIBLE_DEVICES=2
python src/code/T5/infer.py  
        --inp datasets/master/queries-inference/test_formatted_seen_query-seen_doc_test.tsv
        --out datasets/results/$MODEL_NAME/$EXP_TYPE/completions_seen_query-seen_doc_test.mpc
        --doc datasets/master/docs/trec_test.csv
        --ckpt $CHECKPOINT_DIR
        --mdmax_length 48 
        --input_type $EXP_TYPE 
        --model_name $MODEL_NAME
        --beam_size 25 
        --bs $BATCH_SIZE

export CUDA_VISIBLE_DEVICES=3
python src/code/T5/infer.py 
        --inp datasets/master/queries-inference/test_formatted_unseen_query-seen_doc_test.tsv
        --out datasets/results/$MODEL_NAME/$EXP_TYPE/completions_unseen_query-seen_doc_test.mpc
        --doc datasets/master/docs/trec_test.csv
        --ckpt $CHECKPOINT_DIR
        --mdmax_length 48 
        --input_type $EXP_TYPE 
        --model_name $MODEL_NAME
        --beam_size 25 
        --bs $BATCH_SIZE
```


### Experiment 6: RAG - Title,Url and Dense Similar Doc + T5
##### Download Document-Document Similarity File
```

gdown https://drive.google.com/uc\?id\=15nq2d6IvwZmb6v2ZNU1Yw3uYDQ-0m2Ms
unzip similar-docs.zip  -d  datasets/rag
```

##### Step2. Training
```
export MODEL_NAME='t5-small'
export CHECKPOINT_DIR='checkpoints_t5-rag-dense-simdoc'
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Set your gpu ids
export NUM_GPUS=4
mkdir datasets/results/$MODEL_NAME/$EXP_TYPE

export EPOCHS=40 # past experiments 
export BATCH_SIZE=36 # 36 for 64 GB  # 28 for 48GB - training took 40 epochs 2 days with docs

rm -rf $CHECKPOINT_DIR
mkdir $CHECKPOINT_DIR

wandb login ce18e8ae96d72cd78a7a54de441e9657bc0a913d 

export EXP_TYPE='rag_dense_simdoc'

accelerate launch --multi_gpu --num_processes $NUM_GPUS src/code/t5/train.py  --model_dir $CHECKPOINT_DIR --train_data  datasets/master/queries/train.csv --train_doc datasets/master/docs/trec_train.csv --val_data datasets/master/queries/val.csv --val_doc datasets/master/docs/trec_val.csv --bs $BATCH_SIZE   --input_type $EXP_TYPE  --num_epochs $EPOCHS --eval_every 10000000000 --model_name $MODEL_NAME --wandb
```

##### Step3. Inference 
```
export CUDA_VISIBLE_DEVICES=0
export CHECKPOINT_DIR='checkpoints_t5-rag-dense-simdoc'
export MODEL_NAME='t5-small'
export EXP_TYPE='rag_dense_simdoc'
export BATCH_SIZE=36


mkdir datasets/results
mkdir datasets/results/$MODEL_NAME
mkdir datasets/results/$MODEL_NAME/$EXP_TYPE

python src/code/T5/infer.py 
        --inp datasets/master/queries-inference/test_formatted_seen_query-unseen_doc_test.tsv
        --out datasets/results/$MODEL_NAME/$EXP_TYPE/completions_seen_query-unseen_doc_test.mpc 
        --doc datasets/master/docs/trec_test.csv
        --ckpt $CHECKPOINT_DIR
        --mdmax_length 48 
        --input_type $EXP_TYPE 
        --model_name $MODEL_NAME
        --beam_size 25 
        --bs $BATCH_SIZE

export CUDA_VISIBLE_DEVICES=1
python src/code/T5/infer.py 
        --inp datasets/master/queries-inference/test_formatted_unseen_query-unseen_doc_test.tsv
        --out datasets/results/$MODEL_NAME/$EXP_TYPE/completions_unseen_query-unseen_doc_test.mpc
        --doc datasets/master/docs/trec_test.csv
        --ckpt $CHECKPOINT_DIR
        --mdmax_length 48 
        --input_type $EXP_TYPE 
        --model_name $MODEL_NAME
        --beam_size 25 
        --bs $BATCH_SIZE

export CUDA_VISIBLE_DEVICES=2
python src/code/T5/infer.py  
        --inp datasets/master/queries-inference/test_formatted_seen_query-seen_doc_test.tsv
        --out datasets/results/$MODEL_NAME/$EXP_TYPE/completions_seen_query-seen_doc_test.mpc
        --doc datasets/master/docs/trec_test.csv
        --ckpt $CHECKPOINT_DIR
        --mdmax_length 48 
        --input_type $EXP_TYPE 
        --model_name $MODEL_NAME
        --beam_size 25 
        --bs $BATCH_SIZE

export CUDA_VISIBLE_DEVICES=3
python src/code/T5/infer.py 
        --inp datasets/master/queries-inference/test_formatted_unseen_query-seen_doc_test.tsv
        --out datasets/results/$MODEL_NAME/$EXP_TYPE/completions_unseen_query-seen_doc_test.mpc
        --doc datasets/master/docs/trec_test.csv
        --ckpt $CHECKPOINT_DIR
        --mdmax_length 48 
        --input_type $EXP_TYPE 
        --model_name $MODEL_NAME
        --beam_size 25 
        --bs $BATCH_SIZE
```











