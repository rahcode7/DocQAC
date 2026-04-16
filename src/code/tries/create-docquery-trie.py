import pandas as pd 
import numpy as np
from icecream import ic 
from tqdm import tqdm 
import random 
import pickle
import argparse
from utils import QueryCompletion, preprocess, is_empty, load_text_stream, check_and_create_path, get_line_count
import sys
sys.setrecursionlimit(1000)

def is_empty(input_text):
    input_text = input_text.strip()
    if input_text=='' or len(input_text)==0 or len(input_text.split('\t'))<=1:
        return True
    return False

# def init(args):
#     query_completion = QueryCompletion()
#     index = 0
#     #total_count = get_line_count(args.input_file, args.data_limit)
#     #print("creating the main trie using %d queries" % (total_count))
#     #with tqdm(total=total_count) as pbar:
#     for is_valid, input_text in load_text_stream(args.input_file, args.data_limit):
#         pbar.update(1)
#         if not is_valid:
#             break
#         if is_empty(input_text): # or len(input_text)>50:
#             continue
#         query, frequency = input_text.split('\t')
#         print(query,frequency)
#         if int(frequency)<args.threshold:
#             continue
#         query_completion.insert(preprocess(query), int(frequency))
#         print(index)
#         index+=1
#     print("Total queries: ", index)
#     print("saving the main trie")
#     with open(args.output_trie, 'wb') as f:
#         pickle.dump(query_completion, f)
#     print("main trie saved at: ", args.output_trie)

if __name__ == "__main__":

    
    # Read custom orcas dataset - the main one 
    OUTPUT_PATH = "datasets/outputs/docquery-tries/"
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

    # or len(input_text)>50:

    #doc_ids = ['D1000047']
    for ids in tqdm(doc_ids):
        query_completion = QueryCompletion()

        sub_df = orcas_df[orcas_df['docid']==ids]
        #sub_df = orcas_df[orcas_df['docid']=='D1000047']
        print(sub_df)
        # # query_list = sub_df.query.tolist()
        for _,row in sub_df.iterrows():

            #if row['docid'] == 'D1000047':
            print(row['docid'])
            #print(row)
            frequency = row['frequency']    
            query = row['query']         
            print(frequency,query)

            if not pd.isna(query):
                if len(str(query))>100:
                    continue 
                # if int(frequency) < threshold: # Add threshold later
                #     continue
                query_completion.insert(preprocess(query), int(frequency))

        #print("Total queries: ",sub_df.shape[0])

        #print(query_completion.find_completions("justice"))
        trie_path = OUTPUT_PATH + ids  + ".mpc"
        with open(trie_path , 'wb') as f:
            pickle.dump(query_completion, f)    
        
