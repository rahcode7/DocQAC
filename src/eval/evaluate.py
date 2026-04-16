

import json 
import os
import pandas as pd 
import numpy as np
from tqdm import tqdm
import argparse
from sentence_transformers import SentenceTransformer

from pathlib import Path

from metrics import tes,mrr_helper,mean_reciprocal_rank,ndcg_partial_prec,ndcg_partial_rec,get_full_suggestions,bleu_rr,ndcg_alpha,semantic_score_helper
tqdm.pandas()

parser = argparse.ArgumentParser()
parser.add_argument('--data_path') 
args = parser.parse_args()
DATA_PATH = args.data_path
#file_list = ['completions_alpha_0.3_bias_30.0.mpc','completions_unconstrained.mpc']
#file_list = ['completions_seen_query-seen_doc_test.mpc','completions_unseen_query-seen_doc_test.mpc']
#file_list = ['completions_unseen_query-unseen_doc_test.mpc','completions_seen_query-unseen_doc_test.mpc']

# For doc query tries, only2 
#file_list = ['completions_seen_query-seen_doc_test.mpc','completions_unseen_query-seen_doc_test.mpc'] #,'completions_unseen_query-unseen_doc_test.mpc','completions_seen_query-unseen_doc_test.mpc']
#file_list = ['completions_unseen_query-unseen_doc_test.mpc','completions_unseen_query-seen_doc_test.mpc']


# Docquery
file_list = ['completions_seen_query-seen_doc_test.mpc','completions_unseen_query-seen_doc_test.mpc'] #,'completions_unseen_query-unseen_doc_test.mpc','completions_seen_query-unseen_doc_test.mpc']


# #file_list = ['completions_seen_query-seen_doc_test.mpc','completions_unseen_query-seen_doc_test.mpc','completions_unseen_query-unseen_doc_test.mpc']
# #MODEL_LIST = ['t5-rag_sparse_onedoc']#,'t5-title_url_doc','t5-title_url_summary']
# MODEL_LIST = ['doc-tries','docq-tries','global-tries','xc1-prefix','t5-rag_sparse_onedoc','queryblazer-prefix']#,'t5-no-doc']

#MODEL_LIST = ['gpt2-prefix','gpt2-title_url','gpt2-title_url_summary','gpt2-title_url_doc','gpt2-title_url_yake']

#MODEL_LIST = ['t5-prefix','t5-title_url','t5-title_url_doc','t5-title_url_summary'] 
#MODEL_LIST = ['t5-title_url_yake','t5-rag_dense_onedoc'] 


#MODEL_LIST = ['phi3-prefix','phi3-title_url','phi3-title_url_doc','llama3-prefix','llama3-title_url','llama3-title_url_doc']
# SBERT specific 
# Running 
# MODEL_LIST = ['phi3-prefix_only','phi3-prefix_title_doc','phi3-prefix_url_title','phi3-title_url_doc','llama3-prefix_only','llama3-prefix_title_doc','llama3-prefix_url_title','llama3-prefix_title_url_doc']
#MODEL_LIST = ['doc-tries','docq-tries'] 

# 15 Dec
#MODEL_LIST = ['queryblazer-prefix-vocab8192','queryblazer-prefix-n5','queryblazer-prefix','queryblazer-prefix-pruning']
#MODEL_LIST = ['phi3-title_url_doc']
#MODEL_LIST = ['llama3-title_url_summary','llama3-title_url_yake','phi3-title_url_summary','phi3-title_url_yake']

# 6 jan 
#MODEL_LIST = ['queryblazer-title_url','queryblazer-title_url_doc','queryblazer-title_url_summary','queryblazer-title_url_yake']
#op_file = os.path.join(DATA_PATH,"metrics/final","eval_qb.csv")
#MODEL_LIST = ['no_doc'] # after fixing no_doc
#MODEL_LIST = ['docq-tries','docq-tries-title_url','docq-tries-title_url_doc','docq-tries-title_url_yake','docq-tries-title_url_summary','docq-tries-rag_sparse_onedoc','docq-tries-rag_dense_simdoc','docq-tries-rag_dense_onedoc']

# t5 constrained docquery
MODEL_LIST = ['t5const_docq-tries-title_url','t5const_docq-tries-title_url_doc'] #,'t5const_global-tries-title_url_doc','t5const_global-tries-url_doc']
op_file = os.path.join(DATA_PATH,"metrics","eval_t5const_docqtries_1.csv")


# MODEL_LIST = ['global-tries-title_url','global-tries-title_url_doc','global-tries-title_url_summary','global-tries-title_url_yake']
# op_file = os.path.join(DATA_PATH,"metrics/final","eval_globaltries.csv")

# MODEL_LIST = [ 'doc-tries-title_url','doc-tries-title_url_doc','doc-tries-title_url_summary','doc-tries-title_url_yake']
# op_file = os.path.join(DATA_PATH,"metrics/final","eval_doctries.csv")


# MODEL_LIST = ['docq-tries-title_url','docq-tries-title_url_doc','docq-tries-title_url_summary','docq-tries-title_url_yake'
#               ]
# op_file = os.path.join(DATA_PATH,"metrics/final","eval_docqtries.csv")

# MODEL_LIST = ['t5-rag_dense_simdoc','gpt2-rag_dense_onedoc','gpt2-rag_sparse_onedoc','gpt2-rag_dense_simdoc']
# op_file = os.path.join(DATA_PATH,"metrics/final","eval_t5gpt2_2.csv")

#op_file = os.path.join(DATA_PATH,"metrics/final","eval_llm_1.csv")

sb_model = SentenceTransformer("all-MiniLM-L6-v2")


results_df = pd.DataFrame(columns=['model','file','mrr','ndcg_partial_prec','ndcg_partial_rec','tes','bleu_rr','ndcg_alpha','sbertmrr'])
for model_name in MODEL_LIST:

    #print(MODEL_LIST)
    for file in file_list:
        print("Run evals",model_name,file)
        
        data = []
        filepath = os.path.join(DATA_PATH,model_name,file)
        
        filepath = Path(filepath)
        print(filepath)

        # if path exists
        if not os.path.exists(filepath):
            continue

        with open(filepath,encoding="utf-8") as f:
            for row in f:
                data.append(json.loads(row))
        print(len(data))
    
        ndcg_partial_prec_all = []
        ndcg_partial_rec_all = []
        mrr_data_all = []
        bleu_rr_all = []
        ndcg_alpha_all = []
        bs_prec_rr_all = []
        bs_rec_rr_all = []
        bs_f1_rr_all = []
        
        #cnt=0
        sbmrr_data_all = []

        for row in tqdm(data):
            # if cnt>=10:
            #    break 
            # cnt+=1
            gt = row['query']
            pred_list = row['completions']
            prefix = row['prefix']
            # print(gt)
            # print(pred_list,len(pred_list))
            # print(prefix)
            
            if len(pred_list)==1 and not pred_list[0]:
                ndcg_pprec,ndcg_prec,mrr_data,ndcg_alpha_q,bleu_rr_q,sbmrr_data = 0.0,0.0,0.0,0.0,0.0,0.0
            else:
                #print(prefix,gt,pred_list)

                # Merging logic 
                if 'xc' in model_name:
                    print("in xc")
                    suggestion_list = []
                    #print(prefix,pred_list)
                    for suggestion in pred_list:
                        updated_sugg = get_full_suggestions(prefix,suggestion[0])
                        suggestion[0]= updated_sugg
                        suggestion_list.append(suggestion)
                    pred_list = suggestion_list
                    #print(pred_list)
                
                ndcg_pprec = ndcg_partial_prec(gt,pred_list,k=10)
                ndcg_prec = ndcg_partial_rec(gt,pred_list,k=10)
                
                mrr_data = mrr_helper(gt,pred_list)
                ndcg_alpha_q = ndcg_alpha(gt,pred_list,k=10)

                bleu_rr_q = bleu_rr(gt,pred_list)
                
                # SB MRR
                sbmrr_data = semantic_score_helper(sb_model,gt,pred_list)
                
            mrr_data_all.append(mrr_data)
            ndcg_partial_prec_all.append(ndcg_pprec)
            ndcg_partial_rec_all.append(ndcg_prec)
            ndcg_alpha_all.append(ndcg_alpha_q)
            bleu_rr_all.append(bleu_rr_q)
            sbmrr_data_all.append(sbmrr_data)

        mrr_arr = mean_reciprocal_rank(mrr_data_all)
        #mrr_arr_all.append(mrr_arr)

        avg_partial_prec_ndcg = sum(ndcg_partial_prec_all)/len(ndcg_partial_prec_all)
        #avg_partial_prec_ndcg_arr.append(avg_partial_prec_ndcg)

        avg_partial_rec_ndcg = sum(ndcg_partial_rec_all)/len(ndcg_partial_rec_all)
        #avg_partial_rec_ndcg_arr.append(avg_partial_rec_ndcg)

        #print(bleu_rr_all[0:10],len(bleu_rr_all))

        avg_bleu_rr = sum(bleu_rr_all)/len(bleu_rr_all)
        sbmrr_arr = mean_reciprocal_rank(sbmrr_data_all)
        avg_ndcg_alpha = sum(ndcg_alpha_all)/len(ndcg_alpha_all)
        #### TES SCORE
        
        # Slice data for each queries
        # Sort by prefix length

        df = pd.read_json(filepath,lines=True)
    

        df['prefix'] = df['prefix'].fillna('nan')
        df['prefix_len'] = df['prefix'].apply(lambda x: len(x))
        # get unique queries 
        query_list = df['query'].unique().tolist()
        #query_list =['mohonk mountain resort']
        tes_all = []

        for query in tqdm(query_list):
            sub_df = df[df['query']==query]
            sub_df.sort_values(['prefix_len'],ascending=True,inplace=True)
            pred_list = sub_df['completions'].tolist()
            tes_query = tes(query,pred_list)
            tes_all.append(tes_query)

        tes_score_overall = sum(tes_all)/len(tes_all)

        print(mrr_arr,avg_partial_prec_ndcg,avg_partial_rec_ndcg,tes_score_overall,avg_bleu_rr)
        new_row = pd.DataFrame({'file':[file],'model':[model_name],'mrr':[mrr_arr],'ndcg_partial_prec':[avg_partial_prec_ndcg],'ndcg_partial_rec':[avg_partial_rec_ndcg],
                                'tes':[tes_score_overall],'bleu_rr':avg_bleu_rr,'ndcg_alpha':avg_ndcg_alpha,
                                #'bert_score_prec':[avg_bert_score_prec],'bert_score_rec':[avg_bert_score_rec],'bert_score_f1':[avg_bert_score_f1]
                                'sbertmrr' :[sbmrr_arr]
                                })
        print(new_row)
        results_df = pd.concat([results_df, new_row])
        
# print(df)
results_df.to_csv(op_file,index=None)

    


