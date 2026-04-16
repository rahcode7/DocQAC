# Trie Creation

### 1.Create Global Tries


#### Create input dataset for tries
```
python src/code/tries/create-global-tries.py --input_file datasets/master/train.csv 
```

#### 3. Create Training file for MPC
```sh
python   src/code/tries/create_train.py --inp datasets/inputs/all-queries-global.txt --out datasets/inputs/train.mpc
```

```sh
python   src/code/tries/main_trie_creation.py --input_file datasets/inputs/train.mpc --output_trie datasets/outputs/global-tries/main.mpc --threshold 1.0
```
Note: takes in only sentences with freq>=threshold 

### 2.Create docquery tries 
```
python   src/code/mpc/create-docquery-trie.py
```


### 3. Create docngram tries 
```
python   src/code/mpc/create-docngram-trie.py
```

#### Create docngram tries 

```
python   src/code/mpc/create-docngram-trie.py 
```


### Create Suffix Tries for query, docquery and docngram tries

##### Global Tries
```
python   src/code/tries/suffix_trie_creation.py --input_file  datasets/inputs/train.mpc --output_trie datasets/outputs/global-tries/suffix.mpc --suffix_threshold 1
```
Note: takes in only sentences with freq>=suffix_threshold

#####  Suffix tries - docquery tries 
```
python   src/code/mpc/suffix_trie_creation-docq.py --output_trie  datasets/outputs/docquery-tries/suffix --suffix_threshold 1
```

######  Suffix tries - docngram tries  
python   src/code/mpc/suffix_trie_creation-doc.py --output_path  datasets/outputs/docngram-tries/suffix/ --suffix_threshold 1 

#####  Suffix tries - docngram tries - test set 
```
python   src/code/mpc/suffix_trie_creation-doc-test.py 
```

##### Create Test set

```sh
python   src/data-prep/02-test-prefix.py
```

Note: breaks each sentence in test set ```min(n, len(sentence))``` times uniformly randomly through out the sentence while mantaining a ```min_prefix```(default=2) and ```min_suffix```(default=1).


### Step 2 Inference

#### Global tries prediction
```
python  src/code/mpc/parallel_mpc_inference.py --input_file datasets/inputs/test_formatted.tsv --main_trie datasets/inputs/main.mpc --suffix_trie datasets/outputs/global-tries/suffix.mpc --output_file datasets/results/completions.mpc --k_completions 100


python  src/code/tries/parallel_mpc_inference.py --input_file datasets/inputs/test_formatted_seen_query-seen_doc_test.tsv --main_trie datasets/outputs/global-tries/main.mpc --suffix_trie datasets/outputs/global-tries/suffix.mpc --output_file datasets/results/global-tries/completions_seen_query-seen_doc_test.mpc

python  src/code/tries/parallel_mpc_inference.py --input_file datasets/inputs/test_formatted_unseen_query-seen_doc_test.tsv --main_trie datasets/outputs/global-tries/main.mpc --suffix_trie datasets/outputs/global-tries/suffix.mpc --output_file datasets/results/global-tries/completions_unseen_query-seen_doc_test.mpc

python  src/code/tries/parallel_mpc_inference.py --input_file datasets/inputs/test_formatted_seen_query-unseen_doc_test.tsv --main_trie datasets/outputs/global-tries/main.mpc --suffix_trie datasets/outputs/global-tries/suffix.mpc --output_file datasets/results/global-tries/completions_seen_query-unseen_doc_test.mpc

python  src/code/tries/parallel_mpc_inference.py --input_file datasets/inputs/test_formatted_unseen_query-unseen_doc_test.tsv --main_trie datasets/outputs/global-tries/main.mpc --suffix_trie datasets/outputs/global-tries/suffix.mpc --output_file datasets/results/global-tries/completions_unseen_query-unseen_doc_test.mpc
```

##### Global Tries - Document content experiments 

###### Step 1 Get Top 100 completions 
```
python  src/code/mpc/parallel_mpc_inference.py --input_file datasets/inputs/test_formatted_seen_query-seen_doc_test.tsv --main_trie datasets/outputs/global-tries/main.mpc --suffix_trie datasets/outputs/global-tries/suffix.mpc --output_file datasets/inputs/global-tries/top100/completions_seen_query-seen_doc_test.mpc --k_completions 100

python  src/code/mpc/parallel_mpc_inference.py --input_file datasets/inputs/test_formatted_unseen_query-seen_doc_test.tsv --main_trie datasets/outputs/global-tries/main.mpc --suffix_trie datasets/outputs/global-tries/suffix.mpc --output_file datasets/inputs/global-tries/top100/completions_unseen_query-seen_doc_test.mpc --k_completions 100

python  src/code/mpc/parallel_mpc_inference.py --input_file datasets/inputs/test_formatted_seen_query-unseen_doc_test.tsv --main_trie datasets/outputs/global-tries/main.mpc --suffix_trie datasets/outputs/global-tries/suffix.mpc --output_file datasets/inputs/global-tries/top100/completions_seen_query-unseen_doc_test.mpc --k_completions 100

python  src/code/mpc/parallel_mpc_inference.py --input_file datasets/inputs/test_formatted_unseen_query-unseen_doc_test.tsv --main_trie datasets/outputs/global-tries/main.mpc --suffix_trie datasets/outputs/global-tries/suffix.mpc --output_file datasets/inputs/global-tries/top100/completions_unseen_query-unseen_doc_test.mpc --k_completions 100
```

### Step 3 Reranking - Global Tries
##### Title url experiment
```
python src/code/tries/00-reranker-scores.py --completions_path  datasets/inputs/global-tries/top100  --op_path  datasets/results/globaltries-title_url --exp_type tries --context title_url

python src/code/tries/01-reranker-algo.py --ip_path  datasets/results/global-tries-title_url --exp_type tries
```

##### Title url doc experiment
```
python src/code/tries/00-reranker-scores.py --completions_path  datasets/inputs/global-tries/top100  --op_path  datasets/results/global-tries-title_url_doc --exp_type tries --context doc

python src/code/tries/01-reranker-algo.py --ip_path  datasets/results/global-tries-title_url_doc --exp_type tries
```

###### Title url summary experiments
```
python src/code/tries/00-reranker-scores.py --completions_path  datasets/inputs/global-tries/top100  --op_path  datasets/results/global-tries-title_url_summary --exp_type tries --context summary

python src/code/tries/01-reranker-algo.py --ip_path  datasets/results/global-tries-title_url_summary --exp_type tries
```

###### Title url Yake experiments
```
python src/code/tries/00-reranker-scores.py --completions_path  datasets/inputs/global-tries/top100  --op_path  datasets/results/global-tries-title_url_yake --exp_type tries --context yake

python src/code/tries/01-reranker-algo.py --ip_path  datasets/results/global-tries-title_url_yake --exp_type tries
```

###### RAG Sparse One Doc experiments
```
python src/code/tries/00-reranker-scores.py --completions_path  datasets/inputs/global-tries/top100  --op_path  datasets/results/global-tries-rag_sparse_onedoc --exp_type tries --context rag_sparse_onedoc

python src/code/tries/01-reranker-algo.py --ip_path  datasets/results/global-tries-rag_sparse_onedoc --exp_type tries
```

###### RAG Dense  One Doc experiments
```
python src/code/tries/00-reranker-scores.py --completions_path  datasets/inputs/global-tries/top100  --op_path  datasets/results/global-tries-rag_dense_onedoc --exp_type tries --context rag_dense_onedoc

python src/code/tries/01-reranker-algo.py --ip_path  datasets/results/global-tries-rag_dense_onedoc --exp_type tries
```
###### RAG Dense  Similar Doc experiments
```
python src/code/tries/00-reranker-scores.py --completions_path  datasets/inputs/global-tries/top100  --op_path  datasets/results/global-tries-rag_dense_simdoc --exp_type tries --context rag_dense_simdoc

python src/code/tries/01-reranker-algo.py --ip_path  datasets/results/global-tries-rag_dense_simdoc --exp_type tries
```

### Docquery tries
```
python   src/code/mpc/parallel_mpc_inference-docq.py --input_file datasets/inputs/test_formatted_seen_query-seen_doc_test.tsv  --main_trie  datasets/outputs/docquery-tries --output_file datasets/results/docq-tries/completions_seen_query-seen_doc_test_docq.mpc

python   src/code/mpc/parallel_mpc_inference-docq.py --input_file datasets/inputs/test_formatted_unseen_query-seen_doc_test.tsv  --main_trie  datasets/outputs/docquery-tries --output_file datasets/results/docq-tries/completions_unseen_query-seen_doc_test_docq.mpc 

```
### Document Query Tries - Document content experiments 


###### Step 1 Get Top 100 completions 
```
python   src/code/mpc/parallel_mpc_inference-docq.py --input_file datasets/inputs/test_formatted_seen_query-seen_doc_test.tsv  --main_trie  datasets/outputs/docquery-tries --output_file datasets/inputs/docq-tries/top100/completions_seen_query-seen_doc_test_docq.mpc --k_completions 100

python   src/code/mpc/parallel_mpc_inference-docq.py --input_file datasets/inputs/test_formatted_unseen_query-seen_doc_test.tsv  --main_trie  datasets/outputs/docquery-tries --output_file datasets/inputs/docq-tries/top100/completions_unseen_query-seen_doc_test_docq.mpc --k_completions 100

```

#### Step 2 Reranking
##### Title url experiment
```
python src/code/tries/00-reranker-scores.py --completions_path  datasets/inputs/docq-tries/top100  --op_path  datasets/results/docq-tries-title_url --exp_type tries --context title_url  --trie_type docq

python src/code/tries/01-reranker-algo.py --ip_path  datasets/results/docq-tries-title_url --exp_type tries --trie_type docq
```

##### Title url doc experiment
```
python src/code/tries/00-reranker-scores.py --completions_path  datasets/inputs/docq-tries/top100  --op_path  datasets/results/docq-tries-title_url_doc --exp_type tries --context doc --trie_type docq

python src/code/tries/01-reranker-algo.py --ip_path  datasets/results/docq-tries-title_url_doc --exp_type tries --trie_type docq
```

###### Title url summary experiments
```
python src/code/tries/00-reranker-scores.py --completions_path  datasets/inputs/docq-tries/top100  --op_path  datasets/results/docq-tries-title_url_summary --exp_type tries --context summary --trie_type docq

python src/code/tries/01-reranker-algo.py --ip_path  datasets/results/docq-tries-title_url_summary --exp_type tries --trie_type docq
```

###### Title url Yake experiments
```
python src/code/tries/00-reranker-scores.py --completions_path  datasets/inputs/docq-tries/top100  --op_path  datasets/results/docq-tries-title_url_yake --exp_type tries --context yake --trie_type docq

python src/code/tries/01-reranker-algo.py --ip_path  datasets/results/docq-tries-title_url_yake --exp_type tries --trie_type docq
```

###### RAG Sparse One Doc experiments
```
python src/code/tries/00-reranker-scores.py --completions_path  datasets/inputs/docq-tries/top100  --op_path  datasets/results/docq-tries-rag_sparse_onedoc --exp_type tries --context rag_sparse_onedoc --trie_type docq

python src/code/tries/01-reranker-algo.py --ip_path  datasets/results/docq-tries-rag_sparse_onedoc --exp_type tries --trie_type docq
```

###### RAG Dense  One Doc experiments
```

python src/code/tries/00-reranker-scores.py --completions_path  datasets/inputs/docq-tries/top100  --op_path  datasets/results/docq-tries-rag_dense_onedoc --exp_type tries --context rag_dense_onedoc  --trie_type docq

python 01-reranker-algo.py --ip_path  datasets/results/docq-tries-rag_dense_onedoc --exp_type tries --trie_type docq
```

###### RAG Dense  Similar Doc experiments
```
python src/code/tries/00-reranker-scores.py --completions_path  datasets/inputs/docq-tries/top100  --op_path  datasets/results/docq-tries-rag_dense_simdoc --exp_type tries --context rag_dense_simdoc --trie_type docq 

python src/code/tries/01-reranker-algo.py --ip_path  datasets/results/docq-tries-rag_dense_simdoc --exp_type tries --trie_type docq
```
## Docngram tries

##### Seen Doc seen query
```
python   src/code/mpc/parallel_mpc_inference-doc.py --input_file datasets/inputs/test_formatted_seen_query-seen_doc_test.tsv  --main_trie  datasets/outputs/docngram-tries --output_file datasets/results/doc-tries/completions_seen_query-seen_doc_test.mpc
```

##### Seen Doc unseen query
```
python   src/code/mpc/parallel_mpc_inference-doc.py --input_file datasets/inputs/test_formatted_unseen_query-seen_doc_test.tsv  --main_trie  datasets/outputs/docngram-tries --output_file datasets/results/doc-tries/completions_unseen_query-seen_doc_test.mpc 
```

##### UnSeen Doc unseen query
```
python   src/code/mpc/parallel_mpc_inference-doc.py --input_file datasets/inputs/test_formatted_unseen_query-unseen_doc_test.tsv  --main_trie  datasets/outputs/docngram-tries-test-unseenq-ud --output_file datasets/results/doc-tries/completions_unseen_query-unseen_test.mpc 
```

##### UnSeen Doc seen query
```
python   src/code/mpc/parallel_mpc_inference-doc.py --input_file datasets/inputs/test_formatted_seen_query-unseen_doc_test.tsv  --main_trie  datasets/outputs/docngram-tries-test-seenq-ud --output_file datasets/results/doc-tries/completions_seen_query-unseen_doc_test.mpc 
```



### Document Tries - Document content experiments 


###### Step 1 Get Top 100 completions 
##### Seen Doc seen query
```
python   src/code/mpc/parallel_mpc_inference-doc.py --input_file datasets/inputs/test_formatted_seen_query-seen_doc_test.tsv  --main_trie  datasets/outputs/docngram-tries --output_file datasets/inputs/doc-tries/completions_seen_query-seen_doc_test.mpc --k_completions 100
```

##### Seen Doc unseen query
```
python   src/code/mpc/parallel_mpc_inference-doc.py --input_file datasets/inputs/test_formatted_unseen_query-seen_doc_test.tsv  --main_trie  datasets/outputs/docngram-tries --output_file datasets/inputs/doc-tries/completions_unseen_query-seen_doc_test.mpc --k_completions 100
```

##### UnSeen Doc unseen query
```
python   src/code/mpc/parallel_mpc_inference-doc.py --input_file datasets/inputs/test_formatted_unseen_query-unseen_doc_test.tsv  --main_trie  datasets/outputs/docngram-tries-test-unseenq-ud --output_file datasets/inputs/doc-tries/completions_unseen_query-unseen_test.mpc --k_completions 100
```

##### UnSeen Doc seen query
```
python   src/code/mpc/parallel_mpc_inference-doc.py --input_file datasets/inputs/test_formatted_seen_query-unseen_doc_test.tsv  --main_trie  datasets/outputs/docngram-tries-test-seenq-ud --output_file datasets/inputs/doc-tries/completions_seen_query-unseen_doc_test.mpc --k_completions 100
```

#### Step 2 Reranking
##### Title url experiment
```

python src/code/tries/00-reranker-scores.py --completions_path  datasets/inputs/doc-tries/top100  --op_path  datasets/results/doc-tries-title_url --exp_type tries --context title_url  --trie_type doc

python src/code/tries/01-reranker-algo.py --ip_path  datasets/results/doc-tries-title_url --exp_type tries --trie_type doc
```

##### Title url doc experiment
```
python src/code/tries/00-reranker-scores.py --completions_path  datasets/inputs/doc-tries/top100  --op_path  datasets/results/doc-tries-title_url_doc --exp_type tries --context doc --trie_type doc

python src/code/tries/01-reranker-algo.py --ip_path  datasets/results/doc-tries-title_url_doc --exp_type tries --trie_type doc    
```

###### Title url summary experiments
```
python src/code/tries/00-reranker-scores.py --completions_path  datasets/inputs/doc-tries/top100  --op_path  datasets/results/doc-tries-title_url_summary --exp_type tries --context summary --trie_type doc

python src/code/tries/01-reranker-algo.py --ip_path  datasets/results/doc-tries-title_url_summary --exp_type tries --trie_type doc
```

###### Title url Yake experiments
```
python src/code/tries/00-reranker-scores.py --completions_path  datasets/inputs/doc-tries/top100  --op_path  datasets/results/doc-tries-title_url_yake --exp_type tries --context yake --trie_type doc 

python src/code/tries/01-reranker-algo.py --ip_path  datasets/results/doc-tries-title_url_yake --exp_type tries --trie_type doc
```


###### RAG Sparse One Doc experiments
```
python src/code/tries/00-reranker-scores.py --completions_path  datasets/inputs/doc-tries/top100  --op_path  datasets/results/doc-tries-rag_sparse_onedoc --exp_type tries --context rag_sparse_onedoc --trie_type doc

python src/code/tries/01-reranker-algo.py --ip_path  datasets/results/doc-tries-rag_sparse_onedoc --exp_type tries --trie_type doc
```

###### RAG Dense  One Doc experiments
```
python src/code/tries//00-reranker-scores.py --completions_path  datasets/inputs/doc-tries/top100  --op_path  datasets/results/doc-tries-rag_dense_onedoc --exp_type tries --context rag_dense_onedoc --trie_type doc

python src/code/tries/01-reranker-algo.py --ip_path  datasets/results/doc-tries-rag_dense_onedoc --exp_type tries --trie_type doc
```

###### RAG Dense  Similar Doc experiments
```
python src/code/tries//00-reranker-scores.py --completions_path  datasets/inputs/doc-tries/top100  --op_path  datasets/results/doc-tries-rag_dense_simdoc --exp_type tries --context rag_dense_simdoc --trie_type doc

python src/code/tries/01-reranker-algo.py --ip_path  datasets/results/doc-tries-rag_dense_simdoc --exp_type tries --trie_type doc
```
