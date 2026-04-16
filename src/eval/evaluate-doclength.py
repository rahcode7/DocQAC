

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

file_list = ['completions_seen_query-seen_doc_test.mpc','completions_unseen_query-seen_doc_test.mpc','completions_unseen_query-unseen_doc_test.mpc','completions_seen_query-unseen_doc_test.mpc']


# MODEL_LIST = ['docq-tries','doc-tries','global-tries','t5-prefix','t5-title_url','t5-title_url_doc','t5-title_url_summary','t5-rag_sparse_onedoc','t5-rag_dense_onedoc',


# T5 
#MODEL_LIST = ['t5-prefix','gpt2-prefix']

#MODEL_LIST  = ['t5-title_url','t5-title_url_doc','t5-title_url_summary','t5-title_url_yake','t5-rag_dense_onedoc','t5-rag_dense_simdoc','t5-rag_sparse_onedoc']

# GPT2
#MODEL_LIST = ['gpt2-prefix'
#MODEL_LIST = ['gpt2-title_url','gpt2-title_url_doc','gpt2-title_url_summary','gpt2-title_url_yake','gpt2-rag_dense_onedoc','gpt2-rag_sparse_onedoc','gpt2-rag_dense_simdoc']

# Global TRies
#MODEL_LIST = ['global-tries','global-tries-title_url','global-tries-title_url_doc','global-tries-title_url_summary','global-tries-title_url_yake','global-tries-rag_sparse_onedoc','global-tries-rag_dense_onedoc','global-tries-rag_dense_simdoc']

# Doc tries
#MODEL_LIST = ['doc-tries','doc-tries-title_url','doc-tries-title_url_doc','doc-tries-title_url_summary','doc-tries-title_url_yake','doc-tries-rag_dense_onedoc','doc-tries-rag_sparse_onedoc','doc-tries-rag_dense_simdoc']

# Docq tries
#MODEL_LIST = ['docq-tries','docq-tries-title_url','docq-tries-title_url_doc','docq-tries-title_url_summary','docq-tries-title_url_yake','docq-tries-rag_dense_onedoc','docq-tries-rag_sparse_onedoc','docq-tries-rag_dense_simdoc']

# QB
#MODEL_LIST = ['queryblazer-prefix','queryblazer-title_url','queryblazer-title_url_doc','queryblazer-title_url_summary','queryblazer-title_url_yake','queryblazer-rag_sparse_onedoc','queryblazer-rag_dense_onedoc','queryblazer-rag_dense_simdoc']

# phi3
#MODEL_LIST = ['phi3-prefix','phi3-title_url','phi3-title_url_doc','phi3-title_url_summary','phi3-title_url_yake']
#MODEL_LIST = ['phi3-rag_dense_onedoc','phi3-rag_sparse_onedoc','phi3-rag_dense_simdoc']

# llama3
#MODEL_LIST = ['llama3-prefix','llama3-title_url','llama3-title_url_doc','llama3-title_url_summary','llama3-title_url_yake']
MODEL_LIST = ['llama3-title_url_doc']

#MODEL_LIST = ['xc-prefix','xc-title_url','xc-title_url_yake']
#MODEL_LIST = ['xc-title_url_summary','xc-title_url_doc']
#MODEL_LIST = ['llama3-rag_dense_onedoc','llama3-rag_sparse_onedoc','llama3-rag_dense_simdoc']


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

    DATA_PATH = args.data_path  
    #op_file = os.path.join(DATA_PATH,"metrics/doc_len","eval_doclen_docqtries.csv")
    #op_file = os.path.join(DATA_PATH,"metrics/doc_len","eval_doclen_doctries.csv")
    #op_file = os.path.join(DATA_PATH,"metrics/doc_len","eval_doclen_globaltries.csv")
    #op_file = os.path.join(DATA_PATH,"metrics/doc_len","eval_doclen_xc1.csv")
    #op_file = os.path.join(DATA_PATH,"metrics/doc_len","eval_doclen_xc2.csv")
    #op_file = os.path.join(DATA_PATH,"metrics/doc_len","eval_doclen_t5.csv")
    #op_file = os.path.join(DATA_PATH,"metrics/doc_len","eval_doclen_gpt2.csv")
    #op_file = os.path.join(DATA_PATH,"metrics/doc_len","eval_doclen_t5gpt2prefix.csv")
    #op_file = os.path.join(DATA_PATH,"metrics/doc_len","eval_doclen_qb.csv")
    #op_file = os.path.join(DATA_PATH,"metrics/doc_len","eval_doclen_docqtries.csv")
    #op_file = os.path.join(DATA_PATH,"metrics/doc_len","eval_doclen_phi3_rag.csv")
    #op_file = os.path.join(DATA_PATH,"metrics/doc_len","eval_doclen_phi3_nonrag.csv")
    #op_file = os.path.join(DATA_PATH,"metrics/doc_len","eval_doclen_llama3_rag.csv")
    op_file = os.path.join(DATA_PATH,"metrics/doc_len","eval_doclen_llama3_doc.csv")
    #op_file = os.path.join(DATA_PATH,"metrics/doc_len","eval_doclen_llama3_nonrag.csv")
    print(op_file)
    sb_model = SentenceTransformer("all-MiniLM-L6-v2")
    #results_df = pd.DataFrame(columns=['model','file','mrr','ndcg_partial_prec','ndcg_partial_rec','tes','bleu_rr','ndcg_alpha','bert_score_prec','bert_score_rec','bert_score_f1'])
    results_df = pd.DataFrame(columns=['model','file','doc_length_words','data_size','mrr','ndcg_pprec','ndcg_prec','bleu_rr','ndcg_alpha','sbert_mrr','tes'])

    # Read doc length files
    test_df = pd.read_csv("datasets/master/trec_test.csv")
    test_df.head(3)
    print(test_df.columns)

    for model_name in MODEL_LIST:
        print(model_name)
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
            bs_prec_rr_all = []
            bs_rec_rr_all = []
            bs_f1_rr_all = []
            
            #cnt=0
            metric_dict = collections.defaultdict(dict)
            metric_dict['0-200'] = {'mrr_data_all':[],'ndcg_partial_prec_all':[],'ndcg_partial_rec_all':[],'ndcg_alpha_all':[],'bleu_rr_all':[],'sbert_data_all':[]}
            metric_dict['201-500']= {'mrr_data_all':[],'ndcg_partial_prec_all':[],'ndcg_partial_rec_all':[],'ndcg_alpha_all':[],'bleu_rr_all':[],'sbert_data_all':[]}
            metric_dict['501-1500'] = {'mrr_data_all':[],'ndcg_partial_prec_all':[],'ndcg_partial_rec_all':[],'ndcg_alpha_all':[],'bleu_rr_all':[],'sbert_data_all':[]}
            metric_dict['1500+'] = {'mrr_data_all':[],'ndcg_partial_prec_all':[],'ndcg_partial_rec_all':[],'ndcg_alpha_all':[],'bleu_rr_all':[],'sbert_data_all':[]}

            #cnt=0
            for row in tqdm(data):
                # if cnt>1000:
                #    break
                # cnt+=1
                gt = row['query']
                pred_list = row['completions']
                prefix = row['prefix']
                if 'phi3' in model_name or 'llama3' in model_name:
                    #print("yes")
                    docid = row['doc_id'] # for llama phi
                else:
                    docid = row['docid']  # tries,t5 and gpt2
                    
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


                #### Doc length 
                #l_prefix = len(str(prefix))
                l_doc = test_df[test_df['docid']==docid]['body_length'].values
                #print(l_doc)
                ndcg_pprec,ndcg_prec,mrr_data,ndcg_alpha_q,bleu_rr_q,sbert_data  = sample_metric(sb_model,gt,pred_list,k=10)


                if l_doc >= 1 and l_doc <=200:
                    metric_dict['0-200']['mrr_data_all'].append(mrr_data)
                    metric_dict['0-200']['ndcg_partial_prec_all'].append(ndcg_pprec)
                    metric_dict['0-200']['ndcg_partial_rec_all'].append(ndcg_prec)
                    metric_dict['0-200']['ndcg_alpha_all'].append(ndcg_alpha_q)
                    metric_dict['0-200']['bleu_rr_all'].append(bleu_rr_q)
                    metric_dict['0-200']['sbert_data_all'].append(sbert_data)
                    
                elif l_doc > 200 and l_doc <=500:
                    metric_dict['201-500']['mrr_data_all'].append(mrr_data)
                    metric_dict['201-500']['ndcg_partial_prec_all'].append(ndcg_pprec)
                    metric_dict['201-500']['ndcg_partial_rec_all'].append(ndcg_prec)
                    metric_dict['201-500']['ndcg_alpha_all'].append(ndcg_alpha_q)
                    metric_dict['201-500']['bleu_rr_all'].append(bleu_rr_q)
                    metric_dict['201-500']['sbert_data_all'].append(sbert_data)
                    
                elif l_doc > 500 and l_doc <=1500:
                    metric_dict['501-1500']['mrr_data_all'].append(mrr_data)
                    metric_dict['501-1500']['ndcg_partial_prec_all'].append(ndcg_pprec)
                    metric_dict['501-1500']['ndcg_partial_rec_all'].append(ndcg_prec)
                    metric_dict['501-1500']['ndcg_alpha_all'].append(ndcg_alpha_q)
                    metric_dict['501-1500']['bleu_rr_all'].append(bleu_rr_q)
                    metric_dict['501-1500']['sbert_data_all'].append(sbert_data)
                    
                                
                else:
                    metric_dict['1500+']['mrr_data_all'].append(mrr_data)
                    metric_dict['1500+']['ndcg_partial_prec_all'].append(ndcg_pprec)
                    metric_dict['1500+']['ndcg_partial_rec_all'].append(ndcg_prec)
                    metric_dict['1500+']['ndcg_alpha_all'].append(ndcg_alpha_q)
                    metric_dict['1500+']['bleu_rr_all'].append(bleu_rr_q)
                    metric_dict['1500+']['sbert_data_all'].append(sbert_data)

            #print(metric_dict)
            metric_final = collections.defaultdict(dict)

            metric_final['0-200'] = {'mrr_arr':[],'avg_partial_prec_ndcg':[],'avg_partial_rec_ndcg':[],'avg_bleu_rr':[],'avg_ndcg_alpha':[],'data_size':0,'sbert_arr':[]}
            metric_final['201-500']= {'mrr_arr':[],'avg_partial_prec_ndcg':[],'avg_partial_rec_ndcg':[],'avg_bleu_rr':[],'avg_ndcg_alpha':[],'data_size':0,'sbert_arr':[]}
            metric_final['501-1500'] = {'mrr_arr':[],'avg_partial_prec_ndcg':[],'avg_partial_rec_ndcg':[],'avg_bleu_rr':[],'avg_ndcg_alpha':[],'data_size':0,'sbert_arr':[]}
            metric_final['1500+'] = {'mrr_arr':[],'avg_partial_prec_ndcg':[],'avg_partial_rec_ndcg':[],'avg_bleu_rr':[],'avg_ndcg_alpha':[],'data_size':0,'sbert_arr':[]}
             

            metric_final['0-200']['mrr_arr'] = mean_reciprocal_rank(metric_dict['0-200']['mrr_data_all'])
            metric_final['0-200']['sbert_arr'] = mean_reciprocal_rank(metric_dict['0-200']['sbert_data_all'])
            metric_final['0-200']['avg_partial_prec_ndcg'] = sum(metric_dict['0-200']['ndcg_partial_prec_all']) / len(metric_dict['0-200']['ndcg_partial_prec_all'])
            metric_final['0-200']['avg_partial_rec_ndcg'] = sum(metric_dict['0-200']['ndcg_partial_rec_all']) / len(metric_dict['0-200']['ndcg_partial_rec_all'])
            metric_final['0-200']['avg_bleu_rr'] = sum(metric_dict['0-200']['bleu_rr_all']) / len(metric_dict['0-200']['bleu_rr_all'])
            metric_final['0-200']['avg_ndcg_alpha'] = sum(metric_dict['0-200']['ndcg_alpha_all']) / len(metric_dict['0-200']['ndcg_alpha_all'])
            metric_final['0-200']['data_size'] = int(len(metric_dict['0-200']['ndcg_partial_prec_all']))

            
            metric_final['201-500']['mrr_arr'] = mean_reciprocal_rank(metric_dict['201-500']['mrr_data_all'])
            metric_final['201-500']['sbert_arr'] = mean_reciprocal_rank(metric_dict['201-500']['sbert_data_all'])
            metric_final['201-500']['avg_partial_prec_ndcg'] = sum(metric_dict['201-500']['ndcg_partial_prec_all']) / len(metric_dict['201-500']['ndcg_partial_prec_all'])
            metric_final['201-500']['avg_partial_rec_ndcg'] = sum(metric_dict['201-500']['ndcg_partial_rec_all']) / len(metric_dict['201-500']['ndcg_partial_rec_all'])
            metric_final['201-500']['avg_bleu_rr'] = sum(metric_dict['201-500']['bleu_rr_all']) / len(metric_dict['201-500']['bleu_rr_all'])
            metric_final['201-500']['avg_ndcg_alpha'] = sum(metric_dict['201-500']['ndcg_alpha_all']) / len(metric_dict['201-500']['ndcg_alpha_all'])
            metric_final['201-500']['data_size'] = int(len(metric_dict['201-500']['ndcg_partial_prec_all']))

            metric_final['501-1500']['mrr_arr'] = mean_reciprocal_rank(metric_dict['501-1500']['mrr_data_all'])
            metric_final['501-1500']['sbert_arr'] = mean_reciprocal_rank(metric_dict['501-1500']['sbert_data_all'])
            metric_final['501-1500']['avg_partial_prec_ndcg'] = sum(metric_dict['501-1500']['ndcg_partial_prec_all']) / len(metric_dict['501-1500']['ndcg_partial_prec_all'])
            metric_final['501-1500']['avg_partial_rec_ndcg'] = sum(metric_dict['501-1500']['ndcg_partial_rec_all']) / len(metric_dict['501-1500']['ndcg_partial_rec_all'])
            metric_final['501-1500']['avg_bleu_rr'] = sum(metric_dict['501-1500']['bleu_rr_all']) / len(metric_dict['501-1500']['bleu_rr_all'])
            metric_final['501-1500']['avg_ndcg_alpha'] = sum(metric_dict['501-1500']['ndcg_alpha_all']) / len(metric_dict['501-1500']['ndcg_alpha_all'])
            metric_final['501-1500']['data_size'] = int(len(metric_dict['501-1500']['ndcg_partial_prec_all']))


            metric_final['1500+']['mrr_arr'] = mean_reciprocal_rank(metric_dict['1500+']['mrr_data_all'])
            metric_final['1500+']['sbert_arr'] = mean_reciprocal_rank(metric_dict['1500+']['sbert_data_all'])
            metric_final['1500+']['avg_partial_prec_ndcg'] = sum(metric_dict['1500+']['ndcg_partial_prec_all']) / len(metric_dict['1500+']['ndcg_partial_prec_all'])
            metric_final['1500+']['avg_partial_rec_ndcg'] = sum(metric_dict['1500+']['ndcg_partial_rec_all']) / len(metric_dict['1500+']['ndcg_partial_rec_all'])
            metric_final['1500+']['avg_bleu_rr'] = sum(metric_dict['1500+']['bleu_rr_all']) / len(metric_dict['1500+']['bleu_rr_all'])
            metric_final['1500+']['avg_ndcg_alpha'] = sum(metric_dict['1500+']['ndcg_alpha_all']) / len(metric_dict['1500+']['ndcg_alpha_all'])
            metric_final['1500+']['data_size'] = int(len(metric_dict['1500+']['ndcg_partial_prec_all']))
            
            print(metric_final) 

            # Convert to dataframe
            metric_subdf = pd.DataFrame(metric_final)
            print(metric_subdf)
            metric_subdf = metric_subdf.T 
            metric_subdf = metric_subdf.reset_index()
            metric_subdf.columns = ['doc_length_words','mrr','ndcg_pprec','ndcg_prec','bleu_rr','ndcg_alpha','data_size','sbert_mrr']
            metric_subdf['file'] = file
            metric_subdf['model'] = model_name
            metric_subdf = metric_subdf[['model','file','doc_length_words','data_size','mrr','ndcg_pprec','ndcg_prec','bleu_rr','ndcg_alpha','sbert_mrr']]
            
            #### TES SCORE

            tes_dict = collections.defaultdict(dict)
            tes_dict['0-200'] = []
            tes_dict['201-500'] = []
            tes_dict['501-1500'] = []
            tes_dict['1500+'] = []

            df = pd.read_json(filepath,lines=True)
            print(df.columns)

            if 'phi3' in model_name or 'llama3' in model_name:
                print('phi3 & llama renaming docid ')
                df.rename(columns={'doc_id': 'docid'}, inplace=True) # For phi3 llama
            
            unique_pairs = df.drop_duplicates(subset=['query', 'docid'])
            #unique_pairs = df.drop_duplicates(subset=['query', 'doc_id'])

            # Separate into two lists
            query_list = unique_pairs['query'].tolist()
            docid_list = unique_pairs['docid'].tolist()

            #query_list = df['query'].unique().tolist()
            tes_all = []

            for query,docid in tqdm(zip(query_list,docid_list)):
                l_doc = test_df[test_df['docid']==docid]['body_length'].values
                if l_doc >= 1 and l_doc <=200:
                    sub_df = df[df['docid']==docid]
                    pred_list = sub_df['completions'].tolist()
                    tes_query = tes(query,pred_list)
                    # tes_all.append(tes_query)
                    tes_dict['0-200'].append(tes_query)
                elif l_doc >200 and l_doc <=500:
                    sub_df = df[df['docid']==docid]
                    pred_list = sub_df['completions'].tolist()
                    tes_query = tes(query,pred_list)
                    tes_dict['201-500'].append(tes_query)
                elif l_doc >500 and l_doc <=1500:
                    sub_df = df[df['docid']==docid]
                    pred_list = sub_df['completions'].tolist()
                    tes_query = tes(query,pred_list)
                    tes_dict['501-1500'].append(tes_query)
                else:
                    sub_df = df[df['docid']==docid]
                    pred_list = sub_df['completions'].tolist()
                    tes_query = tes(query,pred_list)
                    tes_dict['1500+'].append(tes_query)


            a_0 = sum(tes_dict['0-200'])/len(tes_dict['0-200'])
            a_200 = sum(tes_dict['201-500'])/len(tes_dict['201-500'])
            a_500 = sum(tes_dict['501-1500'])/len(tes_dict['501-1500'])
            a_1500 = sum(tes_dict['1500+'])/len(tes_dict['1500+'])
            metric_subdf['tes'] = [a_0,a_200,a_500,a_1500]
            results_df = pd.concat([results_df,metric_subdf])
            print(results_df)
            
    print(results_df)
    results_df.to_csv(op_file,index=None)
    print("file uploaded")
        


