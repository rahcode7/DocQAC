
### Query Blazer Set Up 

##### 1. Necessary packages
```
sudo apt-get install libboost-all-dev
sudo apt-get install libfst-tools
apt-get install libngram-tools
sudo apt install python3
apt install python3-pip
```


##### 2. QueryBlazer Repo
```
git clone https://github.com/salesforce/QueryBlazer.git && cd QueryBlazer


# first clone all submodules

git submodule update --init --recursive

# install OpenFST & NGram libraries

cd third_party
./install_openfst.sh
```



##### 3.Kenlm
```
wget -O - https://kheafield.com/code/kenlm.tar.gz |tar xz
mkdir kenlm/build
cd kenlm/build
cmake ..
make -j2
cd ../..
```


##### 4. sentence piece
```

cd third_party
git clone https://github.com/google/sentencepiece.git 
cd sentencepiece
mkdir build
cd build
cmake ..
make -j $(nproc)
sudo make install
sudo ldconfig -v
cd ../..
```

##### 5. Build 
```
# back to project root
cd ..


mkdir build && cd build
# requires Boost library serialization module
cmake .. -DCMAKE_CXX_FLAGS=-O2
make -j4
cd ..
```



### Model training 


###### Step 1  Create train.txt 

unzip queryblazer.zip 

##### Step 2 Extract Subword Vocabulary
```
third_party/sentencepiece/build/src/spm_train --model_type bpe --model_prefix subword --input queryblazer/orcas_train.txt --vocab_size 4096 --character_coverage 0.9995
```

##### Step 3  use subword vocabulary and create encoder.fst file
```
cut -f 1 subword.vocab | build/qbz_build_encoder /dev/stdin encoder.fst
```

##### Step 4  Encode Train Corpus
```
build/qbz_encode encoder.fst queryblazer/orcas_train.txt > train.enc
```

##### Step 5 create n-gram language model and save to ngram.arpa
```
third_party/kenlm/build/bin/lmplz --order 8 --discount_fallback  -T . < train.enc > ngram.arpa
```


##### Step 6 convert ngram.arpa to ngram.fst
```
bash script/build_fst_model.sh encoder.fst ngram.arpa ngram.fst
```

##### Step 7 Precompute model 
```
python precompute.py
```

##### Step 8 Inference
```
cut -f1 queryblazer/seen_query-seen_doc_test.tsv | build/qbz_test_queryblazer encoder.fst ngram.fst precomputed.bin /dev/stdin > seen_query-seen_doc_test.completions.tsv

cut -f1 queryblazer/seen_query-unseen_doc_test.tsv | build/qbz_test_queryblazer encoder.fst ngram.fst precomputed.bin /dev/stdin > seen_query-unseen_doc_test.completions.tsv

cut -f1 queryblazer/unseen_query-unseen_doc_test.tsv | build/qbz_test_queryblazer encoder.fst ngram.fst precomputed.bin /dev/stdin > unseen_query-unseen_doc_test.completions.tsv

cut -f1 queryblazer/unseen_query-seen_doc_test.tsv | build/qbz_test_queryblazer encoder.fst ngram.fst precomputed.bin /dev/stdin > unseen_query-seen_doc_test.completions.tsv

```

##### Step 9 Inference with scores
```
python3 inference.py queryblazer/seen_query-seen_doc_test.tsv seen_query-seen_doc-top100.pkl
python3 inference.py queryblazer/seen_query-unseen_doc_test.tsv seen_query-unseen_doc-top100.pkl
python3 inference.py queryblazer/unseen_query-seen_doc_test.tsv unseen_query-seen_doc-top100.pkl
python3 inference.py queryblazer/unseen_query-unseen_doc_test.tsv unseen_query-unseen_doc-top100.pkl
```

##### Evaluations
##### Step 1 read in each query per line and save each prefix/query pair per line, separated by a tab
```
cp queryblazer/seen_query-seen_doc_test.txt queryblazer/seen_query-seen_doc_test.tsv
cp queryblazer/unseen_query-seen_doc_test.txt queryblazer/unseen_query-seen_doc_test.tsv
cp queryblazer/seen_query-unseen_doc_test.txt queryblazer/seen_query-unseen_doc_test.tsv
cp queryblazer/unseen_query-unseen_doc_test.txt queryblazer/unseen_query-unseen_doc_test.tsv
```
###### Step 2 Run Inference
```
cut -f1 queryblazer/seen_query-seen_doc_test.tsv | build/qbz_test_queryblazer encoder.fst ngram.fst precomputed.bin /dev/stdin > seen_query-seen_doc_test.completions.tsv

cut -f1 queryblazer/seen_query-unseen_doc_test.tsv | build/qbz_test_queryblazer encoder.fst ngram.fst precomputed.bin /dev/stdin > seen_query-unseen_doc_test.completions.tsv

cut -f1 queryblazer/unseen_query-unseen_doc_test.tsv | build/qbz_test_queryblazer encoder.fst ngram.fst precomputed.bin /dev/stdin > unseen_query-unseen_doc_test.completions.tsv

cut -f1 queryblazer/unseen_query-seen_doc_test.tsv | build/qbz_test_queryblazer encoder.fst ngram.fst precomputed.bin /dev/stdin > unseen_query-seen_doc_test.completions.tsv
```


###### Step 3 Get files to local
```
scp root@X:QueryBlazer/seen_query-seen_doc_test.completions.tsv .
scp root@X:QueryBlazer/seen_query-unseen_doc_test.completions.tsv .
scp root@X:QueryBlazer/unseen_query-seen_doc_test.completions.tsv .
scp root@X:QueryBlazer/unseen_query-unseen_doc_test.completions.tsv .
```

###### Step 4 Evaluation
```
python data-prep-queryblazer.py
python evaluate.py --data_path datasets/results

```

### Reranking for context experiments


##### 1. Title url experiment
```
python 00-reranker-scores.py --completions_path datasets/inputs/queryblazer/top100  --op_path datasets/results/queryblazer-title_url --exp_type qb --context title_url


python 01-reranker-algo.py --ip_path datasets/results/queryblazer-title_url --exp_type qb
```

##### 2. Title url doc experiment
```
python 00-reranker-scores.py --completions_path datasets/inputs/queryblazer/top100  --op_path datasets/results/queryblazer-title_url_doc --exp_type qb --context doc

python 01-reranker-algo.py --ip_path datasets/results/queryblazer-title_url_doc --exp_type qb
```

###### 3. Title url summary experiments
```
python 00-reranker-scores.py --completions_path datasets/inputs/queryblazer/top100  --op_path datasets/results/queryblazer-title_url_summary --exp_type qb --context summary

python 01-reranker-algo.py --ip_path datasets/results/queryblazer-title_url_summary --exp_type qb
```

###### 4. Title url Yake experiments
```
python 00-reranker-scores.py --completions_path datasets/inputs/queryblazer/top100  --op_path datasets/results/queryblazer-title_url_yake --exp_type qb --context yake


python 01-reranker-algo.py --ip_path datasets/results/queryblazer-title_url_yake --exp_type qb

```

###### RAG Sparse One Doc experiments
```
python 00-reranker-scores.py --completions_path datasets/inputs/queryblazer/top100  --op_path datasets/results/queryblazer-rag_sparse_onedoc --exp_type qb --context rag_sparse_onedoc

python 01-reranker-algo.py --ip_path datasets/results/queryblazer-rag_sparse_onedoc --exp_type qb
```
###### RAG Dense  One Doc experiments
```
python query-auto-suggest-share/src/code/queryblazer/00-reranker-scores.py --completions_path datasets/inputs/queryblazer/top100  --op_path datasets/results/queryblazer-rag_dense_onedoc --exp_type qb --context rag_dense_onedoc

python 01-reranker-algo.py --ip_path datasets/results/queryblazer-rag_dense_onedoc --exp_type qb
```
###### RAG Dense  Similar Doc experiments
```
python query-auto-suggest-share/src/code/queryblazer/00-reranker-scores.py --completions_path datasets/inputs/queryblazer/top100  --op_path datasets/results/queryblazer-rag_dense_simdoc --exp_type qb --context rag_dense_simdoc

python 01-reranker-algo.py --ip_path datasets/results/queryblazer-rag_dense_simdoc --exp_type qb
```
