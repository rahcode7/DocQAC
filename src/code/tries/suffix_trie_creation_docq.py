from tqdm import tqdm
import pickle
import os
import argparse
from utils import QueryCompletion, preprocess, load_text_stream, create_suffixes, get_line_count
from collections import defaultdict
import sys
import pandas as pd 
sys.setrecursionlimit(1000)

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
    parser.add_argument('--output_trie', type=str, required=True,
                        help='path to location to save the trie object.')
    parser.add_argument('--suffix_threshold', type=int, default=0,
                            help='minimum frequency of suffix to be considered for completion.')
    parser.add_argument('--data-limit',
                        type=int,
                        default=-1,
                        help='maximum number of instances to load. -1 to include all.')
    args = parser.parse_args()
    check_and_create_path(args.output_trie)
    
    
    # Read custom orcas dataset - the main one 
    DATA_PATH = "datasets/custom-qac/"
    OUTPUT_PATH = datasets/outputs/docquery-tries/suffix/"
    #orcas_df = pd.read_csv(DATA_PATH + "orcas-qas-5K.csv",sep='\t',header=None)  
    DATA_PATH = "datasets/master/"

    #orcas_df = pd.read_csv(DATA_PATH + "orcas-qas-10K.csv",on_bad_lines='skip')   
    orcas_df = pd.read_csv(DATA_PATH + "train.csv",on_bad_lines='skip')   


    #print(orcas_df.columns)
    #orcas_df.columns = ['qid','query','docid','doc_url']

    #orcas_df = orcas_df.head(5000)
    #print(orcas_df.shape)

    #orcas_df['frequency'] = np.random.randint(1, 500, orcas_df.shape[0])
    orcas_df.rename(columns={'query_count':'frequency'},inplace=True)
    doc_ids = orcas_df['docid'].unique()

    for ids in tqdm(doc_ids):
        query_completion = QueryCompletion()
        suffix_freq = defaultdict(lambda: 0)
        index=0
        sub_df = orcas_df[orcas_df['docid']==ids]
        #sub_df = orcas_df[orcas_df['docid']=='D1000047']
        #print(sub_df)
        # # query_list = sub_df.query.tolist()
        for _,row in sub_df.iterrows():

            #if row['docid'] == 'D1000047':
            #print(row['docid'])
            #print(row)
            frequency = row['frequency']    
            query = row['query']         
            #print(frequency,query)

            suffixes = create_suffixes(preprocess(query))
            for z in suffixes:
                suffix_freq[z] += 1

            if not pd.isna(query):
                if len(str(query))>100:
                    continue 
                # if int(frequency) < threshold: # Add threshold later
                #     continue
                query_completion.insert(preprocess(query), int(frequency))
            index+=1
        
        # print("%d suffixes extracted from %d queries" % (len(suffix_freq), index))
        # print("creating the suffix trie")

        # creating suffix trie
        ctr=0
        for suffix, freq in suffix_freq.items():
            if freq < args.suffix_threshold:
                continue
            #print(suffix,freq)
            query_completion.insert(suffix, freq)
            ctr+=1
        # print("Total suffixes inserted: ", ctr)
        # print("saving the suffix trie")   
        trie_path = OUTPUT_PATH + ids  + ".mpc"
        with open(trie_path, 'wb') as f:
            pickle.dump(query_completion, f)
        #print("suffix trie saved at: ", trie_path)

    
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--input_file', type=str, required=True,
#                         help='file containing the query, frequency pairs.')
#     parser.add_argument('--output_trie', type=str, required=True,
#                         help='path to location to save the trie object.')
#     parser.add_argument('--suffix_threshold', type=int, default=0,
#                             help='minimum frequency of suffix to be considered for completion.')
#     parser.add_argument('--data-limit',
#                         type=int,
#                         default=-1,
#                         help='maximum number of instances to load. -1 to include all.')
#     args = parser.parse_args()
#     check_and_create_path(args.output_trie)
#     init(args)
