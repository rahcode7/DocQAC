import collections
import numpy as np
import pickle 
import pandas as pd 
import os 
import json 
from tqdm import tqdm 
import re
import time 
import argparse

## Rerank last 5 with 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip_path',default="") 
    parser.add_argument('--exp_type',default="") 
    parser.add_argument('--trie_type',default="") 
    
    args = parser.parse_args()

    ipfile_list = ['completions_seen_query-seen_doc_test_scores.mpc','completions_seen_query-unseen_doc_test_scores.mpc','completions_unseen_query-seen_doc_test_scores.mpc','completions_unseen_query-unseen_doc_test_scores.mpc']
    opfile_list = ['completions_seen_query-seen_doc_test.mpc','completions_seen_query-unseen_doc_test.mpc','completions_unseen_query-seen_doc_test.mpc','completions_unseen_query-unseen_doc_test.mpc']

    for file,opfile in zip(ipfile_list,opfile_list):
        #results_list.sort(key=lambda x: x[1],reverse=True)

        if args.trie_type == 'docq' and file in ['completions_seen_query-unseen_doc_test_scores.mpc','completions_unseen_query-unseen_doc_test_scores.mpc']:
            continue

        data = []
        with open(os.path.join(args.ip_path,file),encoding="utf-8") as f:
            for row in f:
                #print(row)
                try:
                    data.append(json.loads(row))
                except json.JSONDecodeError as e:
                    print(row)


        # Logic 1 : 5 Model ranked + 5 reranked based on similarity with context

        output_file = open(os.path.join(args.ip_path,opfile), 'w', encoding='utf-8')
        print(output_file)
        #with open(os.path.join(args.ip_path,file)) as o:
        # for row in tqdm(data):
        #     completions = row['completions']
        #     print(completions)
        #     reranked_list = completions[:5]
        #     #print(reranked_list)
        #     if len(completions)>5:
        #         next_completions = sorted(completions[5:],key=lambda x: x[1],reverse=True)
        #         #print("next completions", next_completions)
        #         reranked_list.extend(next_completions[:5])
        #     #print(reranked_list)
        #     output_file.write(json.dumps({"query": row['query'], "prefix": row['prefix'], "completions": reranked_list}) + "\n")
        
        
        # Rerank top 10 according to similarity scores
        #cnt = 0 
        for row in tqdm(data):
            # cnt+=1
            # if cnt>1:
            #     break 

            completions = row['completions']

            # take top 10
            #print(completions)
            if len(completions)==1 and not completions[0]:
                sortedcompletions = [[]]
            else:
                if args.exp_type=='qb':
                    topcompletions = sorted(completions,key=lambda x: x[2],reverse=False)[0:10] # sort by scores
                    #print(topcompletions)
                elif args.exp_type =='tries':
                    topcompletions = sorted(completions,key=lambda x: x[2],reverse=True)[0:10] # sort by scores
                else:
                    continue
                sortedcompletions = sorted(topcompletions,key=lambda x: x[1],reverse=True) # sort by similarity
            #print(sortedcompletions)
            #print(topcompletions,sortedcompletions)
            #print(sortedcompletions)
            #print(output_file)
            output_file.write(json.dumps({"query": row['query'], "prefix": row['prefix'], "completions": sortedcompletions}) + "\n")
        
        output_file.close()
            

    
