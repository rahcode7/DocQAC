################# Build docContent trie - using up to 5-word-grams with local frequency

import pandas as pd 
#from tries import Trie
import time 
from tqdm import tqdm
import datrie
import re
import collections
import string
from helper import tokenize,generate_n_grams,unique_words,count_n_grams
from utils import QueryCompletion, preprocess, is_empty, load_text_stream, check_and_create_path, get_line_count
import sys
import pickle
sys.setrecursionlimit(1000)
tqdm.pandas()


if __name__ == "__main__":

    # Read docs and split words  
    DATASET_PATH="datasets/"
    DATA_PATH = "datasets/master/"
    OUTPUT_PATH = "datasets/outputs/docngram-tries/"
    
    ## Get TREC documents 
    df = pd.read_csv(DATA_PATH + "trec_train.csv",on_bad_lines='skip')  
    print(f'Number of docs',df.shape[0])
    print(df.head(3))
    print(df.info())

    df = df.dropna(subset=['body'])
    print(df.shape)   

    # Save single trie per document of queries and their popularity
    cnt=0
    for index,row in tqdm(df.iterrows(),total=df.shape[0]):

        query_completion = QueryCompletion()


        docid = row['docid']
        #print(docid)
        # cnt+=1
        # if cnt >=5:
        #     break

        # Get upto 5 grams
        ngram = [1,2,3]
        nname = ['uni','bi','tri']

        ngrams_all = []
        for n in ngram:
            ngrams = generate_n_grams(row['body'], n)
            ngram_counts = count_n_grams(ngrams)
            ngrams_all.extend(ngram_counts.most_common())
        #print("ngrams created",len(ngrams_all))
        query_list = []
        locpop_list = []
        
        for item in ngrams_all:
            #print(item[0],item[1])
            query_list.append(item[0])
            locpop_list.append(item[1])


        #trie = trie_doc_query(query_list,locpop_list)
        for query,frequency in zip(query_list,locpop_list):
            if not pd.isna(query):
                if len(str(query))>100:
                    continue 
                query_completion.insert(preprocess(query), int(frequency))

        trie_path = OUTPUT_PATH + docid  + ".mpc"
        with open(trie_path , 'wb') as f:
            pickle.dump(query_completion, f)   

