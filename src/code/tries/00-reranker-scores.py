from sentence_transformers import SentenceTransformer
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
import csv 
from rag_utils import get_chunks_sparse, rag_sparse_loader, get_chunks_dense, rag_loader_dense, similar_doc_loader,get_similar_docs,get_chunks_similar_dense
pd.set_option('display.max_colwidth', None)


tqdm.pandas()
 
def clean_url(url):
    #https://www.worldwildlife.org/species/directory?direction=desc&sort=extinction_status
    s= re.split(r'[/.?=&\s]+', url)
    s = " ".join(s)
    return s

# similarity(Title + url,pred)
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--completions_path') 
    parser.add_argument('--op_path') 
    parser.add_argument('--exp_type') 
    parser.add_argument('--context')
    parser.add_argument('--trie_type',default="")
    args = parser.parse_args()


   
    Q_PATH = "datasets/inputs"
    DOC_PATH = "datasets/master"
    DOC_SUM_PATH = "datasets/master/summaries"

    TOP100_PATH = args.completions_path
    OUTPUT_PATH = args.op_path 
  
    model = SentenceTransformer("sentence-transformers/msmarco-distilbert-base-v4") # Tuned for cosine similarity - as document length of different sizes   # dim 784
    #model = SentenceTransformer("sentence-transformers/msmarco-MiniLM-L-6-v3")
    print("dim",len(model.encode("hello")))
    top_k=10
    
    # Dataset - Documents 
    if args.context == 'summary':
        doc_df = pd.read_csv(os.path.join(DOC_PATH,'summaries/trec_test_summary_300words.csv'))
    elif args.context == 'yake':
        doc_df = pd.read_csv(os.path.join(DOC_PATH,'yake/trec_test_yake.csv'))
        doc_df['body'] = doc_df['yake']
        doc_df['body'] = doc_df['body'].progress_apply(lambda x : str(x).replace("<eok>","."))
        print(doc_df['body'].head(1))
    else:
        doc_df = pd.read_csv(os.path.join(DOC_PATH,'trec_test.csv'))
        
    
    # or Dataset  -summaries
    #

    # print(doc_df[doc_df['docid']=='D2317482'])

    print(doc_df.columns)
    test_files = ['test_formatted_seen_query-seen_doc_test.tsv','test_formatted_seen_query-unseen_doc_test.tsv','test_formatted_unseen_query-seen_doc_test.tsv','test_formatted_unseen_query-unseen_doc_test.tsv']
    output_files = ['completions_seen_query-seen_doc_test_scores.mpc','completions_seen_query-unseen_doc_test_scores.mpc','completions_unseen_query-seen_doc_test_scores.mpc','completions_unseen_query-unseen_doc_test_scores.mpc']
    pkl_files = ['seen_query-seen_doc-top100.pkl','seen_query-unseen_doc-top100.pkl','unseen_query-seen_doc-top100.pkl','unseen_query-unseen_doc-top100.pkl']
    mpc_files = ['completions_seen_query-seen_doc_test.mpc','completions_seen_query-unseen_doc_test.mpc','completions_unseen_query-seen_doc_test.mpc','completions_unseen_query-unseen_doc_test.mpc']
    qb_files = ['seen_query-seen_doc_test.completions.tsv','seen_query-unseen_doc_test.completions.tsv','unseen_query-seen_doc_test.completions.tsv','unseen_query-unseen_doc_test.completions.tsv']


    # Read queries+prefixes

    if args.context == "rag_dense_simdoc":
        pickle_path="datasets/rag/similar_docs/all/similar_docs.pkl"
        sim_doc_dict = similar_doc_loader(pickle_path)
        embedding, text_splitter = rag_loader_dense(200, 30)
    elif args.context == "rag_dense_onedoc":
        embedding, text_splitter = rag_loader_dense(200, 30)
    elif args.context == "rag_sparse_onedoc":
        text_splitter = rag_sparse_loader()

    for file,output_file,pkl_file,mpc_file,qb_file in zip(test_files,output_files,pkl_files,mpc_files,qb_files):
        print(file)
        # Ignore the following files for docquery tries
        if args.trie_type == 'docq' and mpc_file in ['completions_seen_query-unseen_doc_test.mpc','completions_unseen_query-unseen_doc_test.mpc']:
            continue

        if args.exp_type =='tries':
            # read mpc 
            pred_list = []
            with open(os.path.join(TOP100_PATH,mpc_file),encoding="utf-8") as f:
                for row in f:
                    try:
                        d_row = (json.loads(row))
                        pred_list.append(d_row['completions'])
                    except json.JSONDecodeError as e:
                        print(row)
        elif args.exp_type == 'qb':
            with open(os.path.join(TOP100_PATH,qb_file)) as fd:
                rd = csv.reader(fd, delimiter="\t", quotechar='"')
                pred_list = []
                #cnt=0
                for row in rd:
                    # cnt+=1
                    # if cnt>1:
                    #     break
                    #row[100] #At 100th column, scores tsv start
                    preds = row[0:100]
                    scores = row[101:201]
                    pred_list.append([[p,s] for p,s in zip(preds,scores)])
                #print(pred_list)
        output_file = open(os.path.join(OUTPUT_PATH,output_file), 'w', encoding='utf-8')

        print(len(pred_list))
        q_df = pd.read_csv(os.path.join(Q_PATH,file),sep='\t')
        #q_df = q_df.head(10)
        #print(q_df.head(3))

        # For each prefix
        #i = 0 
        for i,row in tqdm(q_df.iterrows(),total=q_df.shape[0]):
            # if i >10:
            #     break
            # i+=1
            title = doc_df[doc_df['docid']==str(row['docid'])]['heading'].item()
            url = doc_df[doc_df['docid']==str(row['docid'])]['doc_url'].item()
            url = clean_url(url)

            if args.context == 'doc' or args.context == 'summary' or args.context == 'yake':
                body = doc_df[doc_df['docid']==str(row['docid'])]['body'].item()
                context = str(title) + " " + str(url) + " " + str(body)

            elif args.context == "rag_dense_onedoc":
                body = doc_df[doc_df['docid']==str(row['docid'])]['body'].item()
                chunks = get_chunks_dense(row["prefix"], row["docid"], body, embedding, text_splitter)

                final_chunks = ""
                for c in chunks:
                    if c:
                        new_c = " . ".join(c)
                        final_chunks+= new_c + " . "
                context = str(title) + " " + str(url) + str(final_chunks) 

            elif args.context == "rag_dense_simdoc":

                body = doc_df[doc_df['docid']==str(row['docid'])]['body'].item()
                sim_doc_list = get_similar_docs(row['docid'], sim_doc_dict)
                chunks = get_chunks_similar_dense(row["prefix"] ,row["docid"], body, sim_doc_list, embedding,text_splitter, k=40)

                final_chunks = ""
                for c in chunks:
                    if c:
                        new_c = " . ".join(c)
                        final_chunks+= new_c + " . "

                context = str(title) + " " + str(url) + str(final_chunks) 

            elif args.context == "rag_sparse_onedoc":
                body = doc_df[doc_df['docid']==str(row['docid'])]['body'].item()
                chunks = get_chunks_sparse(row["prefix"], body, text_splitter)

                final_chunks = ""
                for c in chunks:
                    if c:
                        new_c = " . ".join(c)
                        final_chunks+= new_c + " . "

                context = str(title) + " " + str(url) + str(final_chunks) 
                #print(context)

            elif args.context == 'title_url':
                context = str(title) + " " + str(url)
            else:
                pass 
            
            encoded_context = model.encode(context)
            #print(len(encoded_context))

            #pred_list = [('endange meaning', 33.947288513183594), ('endange symbol', 34.330657958984375), ('endange', 34.4288215637207), ('endange html', 35.27848815917969), ('endange tag', 35.385658264160156), ('endange a', 35.5563850402832), ('endange x', 35.57109832763672), ('endange mark', 35.6203498840332), ('endange l', 35.667781829833984), ('endange java', 35.6873664855957), ('endange in excel', 35.698486328125), ('endange symbols', 35.86283493041992), ('endange in c', 36.12074279785156), ('endange c', 36.124534606933594), ('endange in html', 36.127105712890625), ('endange in java', 36.19896697998047), ('endange s', 36.22617721557617), ('endange edema', 36.23649215698242), ('endange in matlab', 36.27341079711914), ('endange a', 36.30164337158203)]
            results_list = []
            
            #st= time.time()

            TOP = 50 
            pred_text = [pred[0] for pred in pred_list[i][:TOP]]
            encoded_pred = model.encode(pred_text)
            
            if not pred_list[i]:
               results_list = [[]]
            else:
                scores = model.similarity(encoded_context,encoded_pred).numpy()[0]
                scores_idmax = scores.argsort()[-TOP:][::-1]

                if args.exp_type == 'tries':
                    pred_qb = [pred[1][pred[1].find(':')+1:] for pred in pred_list[i][:TOP]]
                else:
                    pred_qb  = [pred[1] for pred in pred_list[i][:TOP]]
                    
                results_list = [[pred_text[id],float(scores[id]),float(pred_qb[id])] for id in scores_idmax] 
            # et = time.time()
            # print(et-st)
            # print(scores,scores_idmax)

            output_file.write(json.dumps({"query": row['query'], "prefix": row['prefix'], "completions": results_list}) + "\n")
        output_file.close()
        

        

        


