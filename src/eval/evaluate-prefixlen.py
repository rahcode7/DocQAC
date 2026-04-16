

import json 
import os
import pandas as pd 
import numpy as np
from tqdm import tqdm
import argparse
from sentence_transformers import SentenceTransformer
from metrics import tes,mrr_helper,mean_reciprocal_rank,ndcg_partial_prec,ndcg_partial_rec,get_full_suggestions,bleu_rr,ndcg_alpha,semantic_score_helper
import collections 
tqdm.pandas()

file_list = ['completions_seen_query-seen_doc_test.mpc','completions_unseen_query-seen_doc_test.mpc']#,'completions_unseen_query-unseen_doc_test.mpc','completions_seen_query-unseen_doc_test.mpc']
# MODEL_LIST = ['docq-tries','doc-tries','global-tries','t5-prefix','t5-title_url','t5-title_url_doc','t5-title_url_summary','t5-rag_sparse_onedoc','t5-rag_dense_onedoc',


#All 
# MODEL_LIST = ['t5-prefix','t5-title_url','t5-title_url_doc','t5-title_url_summary','t5-title_url_yake','t5-rag_dense_onedoc','t5-rag_dense_simdoc','t5-rag_sparse_onedoc',
#               'gpt2-prefix','gpt2-title_url','gpt2-title_url_doc','gpt2-title_url_summary','gpt2-title_url_yake','gpt2-rag_dense_onedoc','gpt2-rag_sparse_onedoc','gpt2-rag_dense_simdoc',
#               'queryblazer-prefix','queryblazer-title_url','queryblazer-title_url_doc','queryblazer-title_url_summary','queryblazer-title_url_yake','queryblazer-rag_sparse_onedoc','queryblazer-rag_dense_onedoc','queryblazer-rag_dense_simdoc',
#               'phi3-prefix','phi3-title_url','phi3-title_url_doc','phi3-title_url_summary','phi3-title_url_yake',
#               'llama3-prefix','llama3-title_url','llama3_title_url_doc','llama3-title_url_summary','llama3-title_url_yake']



# T5 
#MODEL_LIST = ['t5-prefix','t5-title_url','t5-title_url_doc','t5-title_url_summary','t5-title_url_yake','t5-rag_dense_onedoc','t5-rag_dense_simdoc','t5-rag_sparse_onedoc']

# GPT2
#MODEL_LIST = ['gpt2-prefix','gpt2-title_url','gpt2-title_url_doc','gpt2-title_url_summary','gpt2-title_url_yake','gpt2-rag_dense_onedoc','gpt2-rag_sparse_onedoc','gpt2-rag_dense_simdoc']

# Global TRies
#MODEL_LIST = ['global-tries','global-tries-title_url','global-tries-title_url_doc','global-tries-title_url_summary','global-tries-title_url_yake','global-tries-rag_sparse_onedoc','global-tries-rag_dense_onedoc','global-tries-rag_dense_simdoc']

# Doc tries
#MODEL_LIST = ['doc-tries','doc-tries-title_url','doc-tries-title_url_doc','doc-tries-title_url_summary','doc-tries-title_url_yake','doc-tries-rag_dense_onedoc','doc-tries-rag_sparse_onedoc','doc-tries-rag_dense_simdoc']

# Docq tries
MODEL_LIST = ['docq-tries','docq-tries-title_url','docq-tries-title_url_doc','docq-tries-title_url_summary','docq-tries-title_url_yake','docq-tries-rag_dense_onedoc','docq-tries-rag_sparse_onedoc','docq-tries-rag_dense_simdoc']

# QB
#MODEL_LIST = ['queryblazer-prefix','queryblazer-title_url','queryblazer-title_url_doc','queryblazer-title_url_summary','queryblazer-title_url_yake','queryblazer-rag_sparse_onedoc','queryblazer-rag_dense_onedoc','queryblazer-rag_dense_simdoc']

# phi3
#MODEL_LIST = ['phi3-prefix','phi3-title_url','phi3-title_url_doc','phi3-title_url_summary','phi3-title_url_yake']
 

# llama3
#MODEL_LIST = ['llama3-prefix','llama3-title_url','llama3_title_url_doc','llama3-title_url_summary','llama3-title_url_yake']
 
def sample_metric(sb_model,gt,pred_list,k=10):
    ndcg_pprec = ndcg_partial_prec(gt,pred_list,k)
    ndcg_prec = ndcg_partial_rec(gt,pred_list,k)         
    mrr_data = mrr_helper(gt,pred_list)
    ndcg_alpha_q = ndcg_alpha(gt,pred_list,k)
    sbert_data = semantic_score_helper(sb_model,gt,pred_list)
    bleu_rr_q = bleu_rr(gt,pred_list)

    return ndcg_pprec,ndcg_prec,mrr_data,ndcg_alpha_q,bleu_rr_q,sbert_data
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path') 
    args = parser.parse_args()

    DATA_PATH = args.data_path  #
    
    #op_file = os.path.join(DATA_PATH,"metrics/prefix_len","eval_prefixlen_doctries.csv")
    #op_file = os.path.join(DATA_PATH,"metrics/prefix_len","eval_prefixlen_globaltries.csv")
    op_file = os.path.join(DATA_PATH,"metrics/prefix_len","eval_prefixlen_docqtries.csv")
    #op_file = os.path.join(DATA_PATH,"metrics/prefix_len","eval_prefixlen_t5.csv")
    #op_file = os.path.join(DATA_PATH,"metrics/prefix_len","eval_prefixlen_gpt2.csv")
    #op_file = os.path.join(DATA_PATH,"metrics/prefix_len","eval_prefixlen_qb.csv")
    #op_file = os.path.join(DATA_PATH,"metrics/prefix_len","eval_prefixlen_phi3_nonrag.csv")
    #op_file = os.path.join(DATA_PATH,"metrics/prefix_len","eval_prefixlen_llama3_nonrag.csv")
    print(op_file)

    sb_model = SentenceTransformer("all-MiniLM-L6-v2")

    #results_df = pd.DataFrame(columns=['model','file','mrr','ndcg_partial_prec','ndcg_partial_rec','tes','bleu_rr','ndcg_alpha','bert_score_prec','bert_score_rec','bert_score_f1'])
    results_df = pd.DataFrame(columns=['model','file','prefix_length','data_size','mrr','ndcg_pprec','ndcg_prec','bleu_rr','ndcg_alpha','sbert_mrr'])

    for model_name in MODEL_LIST:
        #print(MODEL_LIST)
        for file in file_list:
            print("Run evals",model_name,file)
            
            data = []
            filepath = os.path.join(DATA_PATH,model_name,file)

            # if path exists
            if not os.path.exists(filepath):
                continue

            with open(filepath,encoding="utf-8") as f:
                for row in f:
                    data.append(json.loads(row))
        
            ndcg_partial_prec_all = []
            ndcg_partial_rec_all = []
            mrr_data_all = []
            bleu_rr_all = []
            ndcg_alpha_all = []
            # bs_prec_rr_all = []
            # bs_rec_rr_all = []
            # bs_f1_rr_all = []
            
            #cnt=0
            metric_dict = collections.defaultdict(dict)
            metric_dict['1-5'] = {'mrr_data_all':[],'ndcg_partial_prec_all':[],'ndcg_partial_rec_all':[],'ndcg_alpha_all':[],'bleu_rr_all':[],'sbert_data_all':[]}
            metric_dict['6-10']= {'mrr_data_all':[],'ndcg_partial_prec_all':[],'ndcg_partial_rec_all':[],'ndcg_alpha_all':[],'bleu_rr_all':[],'sbert_data_all':[]}
            metric_dict['11-15'] = {'mrr_data_all':[],'ndcg_partial_prec_all':[],'ndcg_partial_rec_all':[],'ndcg_alpha_all':[],'bleu_rr_all':[],'sbert_data_all':[]}
            metric_dict['16-20'] = {'mrr_data_all':[],'ndcg_partial_prec_all':[],'ndcg_partial_rec_all':[],'ndcg_alpha_all':[],'bleu_rr_all':[],'sbert_data_all':[]}
            metric_dict['20+'] = {'mrr_data_all':[],'ndcg_partial_prec_all':[],'ndcg_partial_rec_all':[],'ndcg_alpha_all':[],'bleu_rr_all':[],'sbert_data_all':[]}
             
            for row in tqdm(data):
                # if cnt>=1000:
                #    break 
                # cnt+=1
                gt = row['query']
                pred_list = row['completions']
                prefix = row['prefix']
                #print("prefix",prefix)
                #print(gt,pred_list)

                # Merging logic for xc
                if 'xc' in model_name:
                    suggestion_list = []
                    #print(prefix,pred_list)
                    for suggestion in pred_list:
                        updated_sugg = get_full_suggestions(prefix,suggestion[0])
                        suggestion[0]= updated_sugg
                        suggestion_list.append(suggestion)
                    pred_list = suggestion_list
                    #print(pred_list)


                #### Prefix length 
                l_prefix = len(str(prefix))
                ndcg_pprec,ndcg_prec,mrr_data,ndcg_alpha_q,bleu_rr_q,sbert_data  = sample_metric(sb_model,gt,pred_list,k=10)

                if l_prefix >= 1 and l_prefix <=5:
                    metric_dict['1-5']['mrr_data_all'].append(mrr_data)
                    metric_dict['1-5']['ndcg_partial_prec_all'].append(ndcg_pprec)
                    metric_dict['1-5']['ndcg_partial_rec_all'].append(ndcg_prec)
                    metric_dict['1-5']['ndcg_alpha_all'].append(ndcg_alpha_q)
                    metric_dict['1-5']['bleu_rr_all'].append(bleu_rr_q)
                    metric_dict['1-5']['sbert_data_all'].append(sbert_data)
                    
                elif l_prefix > 5 and l_prefix <=10:
                    metric_dict['6-10']['mrr_data_all'].append(mrr_data)
                    metric_dict['6-10']['ndcg_partial_prec_all'].append(ndcg_pprec)
                    metric_dict['6-10']['ndcg_partial_rec_all'].append(ndcg_prec)
                    metric_dict['6-10']['ndcg_alpha_all'].append(ndcg_alpha_q)
                    metric_dict['6-10']['bleu_rr_all'].append(bleu_rr_q)
                    metric_dict['6-10']['sbert_data_all'].append(sbert_data)
                    
                    
                elif l_prefix > 10 and l_prefix <=15:
                    metric_dict['11-15']['mrr_data_all'].append(mrr_data)
                    metric_dict['11-15']['ndcg_partial_prec_all'].append(ndcg_pprec)
                    metric_dict['11-15']['ndcg_partial_rec_all'].append(ndcg_prec)
                    metric_dict['11-15']['ndcg_alpha_all'].append(ndcg_alpha_q)
                    metric_dict['11-15']['bleu_rr_all'].append(bleu_rr_q)
                    metric_dict['11-15']['sbert_data_all'].append(sbert_data)
                    
                
                elif l_prefix > 15 and l_prefix <=20:
                    metric_dict['16-20']['mrr_data_all'].append(mrr_data)
                    metric_dict['16-20']['ndcg_partial_prec_all'].append(ndcg_pprec)
                    metric_dict['16-20']['ndcg_partial_rec_all'].append(ndcg_prec)
                    metric_dict['16-20']['ndcg_alpha_all'].append(ndcg_alpha_q)
                    metric_dict['16-20']['bleu_rr_all'].append(bleu_rr_q)
                    metric_dict['16-20']['sbert_data_all'].append(sbert_data)
                    
                
                else:
                    metric_dict['20+']['mrr_data_all'].append(mrr_data)
                    metric_dict['20+']['ndcg_partial_prec_all'].append(ndcg_pprec)
                    metric_dict['20+']['ndcg_partial_rec_all'].append(ndcg_prec)
                    metric_dict['20+']['ndcg_alpha_all'].append(ndcg_alpha_q)
                    metric_dict['20+']['bleu_rr_all'].append(bleu_rr_q)
                    metric_dict['20+']['sbert_data_all'].append(sbert_data)
                    

            #print(metric_dict)
            metric_final = collections.defaultdict(dict)

            metric_final['1-5'] = {'mrr_arr':[],'avg_partial_prec_ndcg':[],'avg_partial_rec_ndcg':[],'avg_bleu_rr':[],'avg_ndcg_alpha':[],'data_size':0,'sbert_arr':[]}
            metric_final['6-10']= {'mrr_arr':[],'avg_partial_prec_ndcg':[],'avg_partial_rec_ndcg':[],'avg_bleu_rr':[],'avg_ndcg_alpha':[],'data_size':0,'sbert_arr':[]}
            metric_final['11-15'] = {'mrr_arr':[],'avg_partial_prec_ndcg':[],'avg_partial_rec_ndcg':[],'avg_bleu_rr':[],'avg_ndcg_alpha':[],'data_size':0,'sbert_arr':[]}
            metric_final['16-20'] ={'mrr_arr':[],'avg_partial_prec_ndcg':[],'avg_partial_rec_ndcg':[],'avg_bleu_rr':[],'avg_ndcg_alpha':[],'data_size':0,'sbert_arr':[]}
            metric_final['20+'] = {'mrr_arr':[],'avg_partial_prec_ndcg':[],'avg_partial_rec_ndcg':[],'avg_bleu_rr':[],'avg_ndcg_alpha':[],'data_size':0,'sbert_arr':[]}
             

            metric_final['1-5']['mrr_arr'] = mean_reciprocal_rank(metric_dict['1-5']['mrr_data_all'])
            metric_final['1-5']['sbert_arr'] = mean_reciprocal_rank(metric_dict['1-5']['sbert_data_all'])
            #if not metric_dict['1-5']['ndcg_partial_prec_all']:
            metric_final['1-5']['avg_partial_prec_ndcg'] = sum(metric_dict['1-5']['ndcg_partial_prec_all']) / len(metric_dict['1-5']['ndcg_partial_prec_all'])
            #else:
            #metric_final['1-5']['avg_partial_prec_ndcg'] = 0 

            
            metric_final['1-5']['avg_partial_rec_ndcg'] = sum(metric_dict['1-5']['ndcg_partial_rec_all']) / len(metric_dict['1-5']['ndcg_partial_rec_all'])
            metric_final['1-5']['avg_bleu_rr'] = sum(metric_dict['1-5']['bleu_rr_all']) / len(metric_dict['1-5']['bleu_rr_all'])
            
            metric_final['1-5']['avg_ndcg_alpha'] = sum(metric_dict['1-5']['ndcg_alpha_all']) / len(metric_dict['1-5']['ndcg_alpha_all'])
            metric_final['1-5']['data_size'] = int(len(metric_dict['1-5']['ndcg_partial_prec_all']))
            
            metric_final['6-10']['mrr_arr'] = mean_reciprocal_rank(metric_dict['6-10']['mrr_data_all'])
            metric_final['6-10']['sbert_arr'] = mean_reciprocal_rank(metric_dict['6-10']['sbert_data_all'])
            metric_final['6-10']['avg_partial_prec_ndcg'] = sum(metric_dict['6-10']['ndcg_partial_prec_all']) / len(metric_dict['6-10']['ndcg_partial_prec_all'])
            metric_final['6-10']['avg_partial_rec_ndcg'] = sum(metric_dict['6-10']['ndcg_partial_rec_all']) / len(metric_dict['6-10']['ndcg_partial_rec_all'])
            metric_final['6-10']['avg_bleu_rr'] = sum(metric_dict['6-10']['bleu_rr_all']) / len(metric_dict['6-10']['bleu_rr_all'])
            metric_final['6-10']['avg_ndcg_alpha'] = sum(metric_dict['6-10']['ndcg_alpha_all']) / len(metric_dict['6-10']['ndcg_alpha_all'])
            metric_final['6-10']['data_size'] = int(len(metric_dict['6-10']['ndcg_partial_prec_all']))

            metric_final['11-15']['mrr_arr'] = mean_reciprocal_rank(metric_dict['11-15']['mrr_data_all'])
            metric_final['11-15']['sbert_arr'] = mean_reciprocal_rank(metric_dict['11-15']['sbert_data_all'])
            metric_final['11-15']['avg_partial_prec_ndcg'] = sum(metric_dict['11-15']['ndcg_partial_prec_all']) / len(metric_dict['11-15']['ndcg_partial_prec_all'])
            metric_final['11-15']['avg_partial_rec_ndcg'] = sum(metric_dict['11-15']['ndcg_partial_rec_all']) / len(metric_dict['11-15']['ndcg_partial_rec_all'])
            metric_final['11-15']['avg_bleu_rr'] = sum(metric_dict['11-15']['bleu_rr_all']) / len(metric_dict['11-15']['bleu_rr_all'])
            metric_final['11-15']['avg_ndcg_alpha'] = sum(metric_dict['11-15']['ndcg_alpha_all']) / len(metric_dict['11-15']['ndcg_alpha_all'])
            metric_final['11-15']['data_size'] = int(len(metric_dict['11-15']['ndcg_partial_prec_all']))

            metric_final['16-20']['mrr_arr'] = mean_reciprocal_rank(metric_dict['16-20']['mrr_data_all'])
            metric_final['16-20']['sbert_arr'] = mean_reciprocal_rank(metric_dict['16-20']['sbert_data_all'])
            metric_final['16-20']['avg_partial_prec_ndcg'] = sum(metric_dict['16-20']['ndcg_partial_prec_all']) / len(metric_dict['16-20']['ndcg_partial_prec_all'])
            metric_final['16-20']['avg_partial_rec_ndcg'] = sum(metric_dict['16-20']['ndcg_partial_rec_all']) / len(metric_dict['16-20']['ndcg_partial_rec_all'])
            metric_final['16-20']['avg_bleu_rr'] = sum(metric_dict['16-20']['bleu_rr_all']) / len(metric_dict['16-20']['bleu_rr_all'])
            metric_final['16-20']['avg_ndcg_alpha'] = sum(metric_dict['16-20']['ndcg_alpha_all']) / len(metric_dict['16-20']['ndcg_alpha_all'])
            metric_final['16-20']['data_size'] = int(len(metric_dict['16-20']['ndcg_partial_prec_all']))


            # if len(metric_dict['16-20']['ndcg_partial_prec_all'])!=0: 
            #     # print("not empty")
            #     # print(len(metric_dict['16-20']['ndcg_partial_prec_all']))
            #     # print(type(metric_dict['16-20']['ndcg_partial_prec_all']))
            #     # print(metric_dict['16-20']['ndcg_partial_prec_all'])
            #     metric_final['16-20']['avg_partial_prec_ndcg'] = sum(metric_dict['16-20']['ndcg_partial_prec_all']) / len(metric_dict['16-20']['ndcg_partial_prec_all'])
            # else:
            #     metric_final['16-20']['avg_partial_prec_ndcg'] = 0.0
            
            # if len(metric_dict['16-20']['ndcg_partial_rec_all'])!=0: 
            #     metric_final['16-20']['avg_partial_rec_ndcg'] = sum(metric_dict['16-20']['ndcg_partial_rec_all']) / len(metric_dict['16-20']['ndcg_partial_rec_all'])
            # else:
            #     metric_final['16-20']['avg_partial_rec_ndcg'] = 0.0

            # if len(metric_dict['16+']['bleu_rr_all'])!=0:
            #     metric_final['16-20']['avg_bleu_rr'] = mean_reciprocal_rank(metric_dict['16+']['bleu_rr_all'])
            # else:
            #     metric_final['16-20']['avg_bleu_rr']= 0.0

            # if len(metric_dict['16-20']['ndcg_alpha_all'])!=0: 
            #     metric_final['16-20']['avg_ndcg_alpha'] = sum(metric_dict['16-20']['ndcg_alpha_all']) / len(metric_dict['16-20']['ndcg_alpha_all'])
            # else:
            #     metric_final['16-20']['avg_ndcg_alpha'] = 0.0
            # metric_final['16-20']['data_size'] = int(len(metric_dict['16-20']['ndcg_partial_prec_all']))


            metric_final['20+']['mrr_arr'] = mean_reciprocal_rank(metric_dict['20+']['mrr_data_all'])
            metric_final['20+']['sbert_arr'] = mean_reciprocal_rank(metric_dict['20+']['sbert_data_all'])
            metric_final['20+']['avg_partial_prec_ndcg'] = sum(metric_dict['20+']['ndcg_partial_prec_all']) / len(metric_dict['20+']['ndcg_partial_prec_all'])
            metric_final['20+']['avg_partial_rec_ndcg'] = sum(metric_dict['20+']['ndcg_partial_rec_all']) / len(metric_dict['20+']['ndcg_partial_rec_all'])
            metric_final['20+']['avg_bleu_rr'] =  sum(metric_dict['20+']['bleu_rr_all']) / len(metric_dict['20+']['bleu_rr_all'])
            metric_final['20+']['avg_ndcg_alpha'] = sum(metric_dict['20+']['ndcg_alpha_all']) / len(metric_dict['20+']['ndcg_alpha_all'])
            metric_final['20+']['data_size'] = int(len(metric_dict['20+']['ndcg_partial_prec_all']))
    
                     
            print(metric_final)

            # Convert to dataframe
            metric_subdf = pd.DataFrame(metric_final)
            print(metric_subdf)
            metric_subdf = metric_subdf.T 
            metric_subdf = metric_subdf.reset_index()
            metric_subdf.columns = ['prefix_length','mrr','ndcg_pprec','ndcg_prec','bleu_rr','ndcg_alpha','data_size','sbert_mrr']
            metric_subdf['file'] = file
            metric_subdf['model'] = model_name
            metric_subdf = metric_subdf[['model','file','prefix_length','data_size','mrr','ndcg_pprec','ndcg_prec','bleu_rr','ndcg_alpha','sbert_mrr']]
            results_df = pd.concat([results_df,metric_subdf])
 
            
            
            #### TES SCORE
            
            # Slice data for each queries
            # Sort by prefix length

            
            # print(df.head(3))
            # print(df.info())


            # if df['query'].shape[0] != df['prefix'].shape[0] :
            #     tes_score_overall = 0.0
            # else:
            # df = pd.read_json(filepath,lines=True)
            # df['prefix'] = df['prefix'].fillna('nan')
            # df['prefix_len'] = df['prefix'].apply(lambda x: len(x))
            # # get unique queries 
            # query_list = df['query'].unique().tolist()
            # tes_all = []


            # tes_dict = collections.defaultdict(dict)
            # tes_dict['1-5'] = []
            # tes_dict['6-10'] = []
            # tes_dict['11-15'] = []
            # tes_dict['16-20'] = []
            # tes_dict['20+'] = [] 

            # for query in tqdm(query_list):
            #     #print("query:",query)
            #     sub_df = df[df['query']==query]
            #     sub_df.sort_values(['prefix_len'],ascending=True,inplace=True)
            #     #print(sub_df['completions'])
            #     #print(sub_df)
            #     pred_list = sub_df['completions'].tolist()

            #     #print(pred_list)
            #     tes_query = tes(query,pred_list)
            #     #print(f"query {query} tes_score {tes_query} pred_list {pred_list}")

            #     tes_all.append(tes_query)
            #     print(tes_all)
            # tes_score_overall = sum(tes_all)/len(tes_all)

            #print(mrr_arr,avg_partial_prec_ndcg,avg_partial_rec_ndcg,tes_score_overall,avg_bleu_rr)
            # new_row = pd.DataFrame({'file':[file],'model':[model_name],'mrr':[mrr_arr],'ndcg_partial_prec':[avg_partial_prec_ndcg],'ndcg_partial_rec':[avg_partial_rec_ndcg],
            #                         'tes':[tes_score_overall],'bleu_rr':avg_bleu_rr,'ndcg_alpha':avg_ndcg_alpha,
            #                         'bert_score_prec':[avg_bert_score_prec],'bert_score_rec':[avg_bert_score_rec],'bert_score_f1':[avg_bert_score_f1]
            #                         })
            # print(new_row)
            # results_df = pd.concat([results_df, new_row])
            
    # print(df)
    results_df.to_csv(op_file,index=None)

        


