from tqdm import tqdm
import pickle
import os
import argparse
from utils import QueryCompletion, preprocess, load_text_stream, create_suffixes, get_line_count
from collections import defaultdict
import sys
import pandas as pd 
from helper import tokenize,generate_n_grams,unique_words,count_n_grams
sys.setrecursionlimit(5000)

# check whether base_path to a file exists or not and create if it doesn't
def check_and_create_path(file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))

def is_empty(input_text):
    input_text = input_text.strip()
    if input_text=='' or len(input_text)==0 or len(input_text.split('\t'))<=1:
        return True
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--input_file', type=str, required=True,
    #                     help='file containing the query, frequency pairs.')
    parser.add_argument('--output_path', type=str, required=True,
                        help='path to location to save the trie object.')
    parser.add_argument('--suffix_threshold', type=int, default=0,
                            help='minimum frequency of suffix to be considered for completion.')
    parser.add_argument('--data-limit',
                        type=int,
                        default=-1,
                        help='maximum number of instances to load. -1 to include all.')
    args = parser.parse_args()
    check_and_create_path(args.output_path)
    
    
   
    DATASET_PATH="datasets/"
    DATA_PATH = "datasets/master/"
    OUTPUT_PATH = args.output_path

    
    ## 1. Get TREC documents 
    df = pd.read_csv(DATA_PATH + "trec_train.csv",on_bad_lines='skip')  
    print(f'Number of docs',df.shape[0])
    print(df.head(3))
    print(df.info())

    df = df.dropna(subset=['body'])
    print(df.shape)   

    ## 2. Create suffix tries for each documents 
    cnt=0
    for index,row in tqdm(df.iterrows(),total=df.shape[0]):
        query_completion = QueryCompletion()
        suffix_freq = defaultdict(lambda: 0)

        docid = row['docid']
        #print(docid)

        # cnt+=1
        # if cnt >=2:
        # 

        # Get upto 5 grams
        ngram = [1,2,3] #4] #,4,5]
        nname = ['uni','bi','tri'] #,'four'] # ,'four','five']

        ngrams_all = []
        for n in ngram:
            ngrams = generate_n_grams(row['body'], n)
            ngram_counts = count_n_grams(ngrams)
            ngrams_all.extend(ngram_counts.most_common())
        
        query_list = []
        locpop_list = []
        
        for item in ngrams_all:
            #print(item[0],item[1])
            query_list.append(item[0])
            locpop_list.append(item[1])


        for query,frequency  in zip(query_list,locpop_list):

            suffixes = create_suffixes(preprocess(query))
            for z in suffixes:
                suffix_freq[z] += 1
            index+=1

        # creating suffix trie

        ctr=0
        for suffix, freq in suffix_freq.items():
            if ctr >=10000:
                break
            #print("Total suffixes ", len(suffix))
            if freq < args.suffix_threshold:
                continue
            #print(suffix,freq)
            query_completion.insert(suffix, freq)
            ctr+=1
        #print("Total suffixes inserted: ", ctr)
        # print("saving the suffix trie")   
        trie_path = OUTPUT_PATH + "/" + docid  + ".mpc"

        try:
            with open(trie_path, 'wb') as f:
                pickle.dump(query_completion, f)
        except Exception:
            continue
            
