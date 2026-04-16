import json 
import os
import pandas as pd 
import numpy as np
from tqdm import tqdm
import argparse
from sentence_transformers import SentenceTransformer
from metrics import tes,mrr_helper,mean_reciprocal_rank,ndcg_partial_prec,ndcg_partial_rec,get_full_suggestions,bleu_rr,ndcg_alpha,semantic_score_helper
import collections 

# --- Helper Functions ---
def sample_metric(sb_model,gt,pred_list,k=10):
    ndcg_pprec = ndcg_partial_prec(gt,pred_list,k)
    ndcg_prec = ndcg_partial_rec(gt,pred_list,k)      
    mrr_data = mrr_helper(gt,pred_list)
    ndcg_alpha_q = ndcg_alpha(gt,pred_list,k)
    sbert_data = semantic_score_helper(sb_model,gt,pred_list)
    bleu_rr_q = bleu_rr(gt,pred_list)
    return ndcg_pprec,ndcg_prec,mrr_data,ndcg_alpha_q,bleu_rr_q,sbert_data

def calculate_average(metric_list):
    """Safely calculate the average, returning 0.0 for an empty list."""
    if not metric_list:
        return 0.0
    return sum(metric_list) / len(metric_list)

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path') 
    args = parser.parse_args()

    DATA_PATH = args.data_path  
    
    # --- Configuration ---
    file_list = ['completions_seen_query-seen_doc_test.mpc','completions_unseen_query-seen_doc_test.mpc','completions_unseen_query-unseen_doc_test.mpc','completions_seen_query-unseen_doc_test.mpc']
    # DocQ Tries
    # MODEL_LIST = ['docq-tries-prefix','docq-tries-rag_dense_onedoc','docq-tries-rag_dense_simdoc','docq-tries-rag_sparse_onedoc','docq-tries-title_url','docq-tries-title_url_doc','docq-tries-title_url_summary','docq-tries-title_url_yake']
    # op_file = os.path.join(DATA_PATH, "metrics", "eval_doclen_docq_overall.csv")

    # Doc Tries
    # MODEL_LIST = ['doc-tries-prefix','doc-tries-rag_dense_onedoc','doc-tries-rag_dense_simdoc','doc-tries-rag_sparse_onedoc','doc-tries-title_url','doc-tries-title_url_doc','doc-tries-title_url_summary','doc-tries-title_url_yake']
    # op_file = os.path.join(DATA_PATH, "metrics", "eval_doclen_doc_overall.csv")

    # Global
    # MODEL_LIST = ['global-tries-prefix','global-tries-title_url','global-tries-title_url_doc','global-tries-title_url_summary','global-tries-title_url_yake','global-tries-rag_dense_sim_doc','global-tries-rag_dense_one_doc','global-tries-rag_sparse_one_doc']
    # op_file = os.path.join(DATA_PATH, "metrics", "eval_doclen_global_overall.csv")

    # Phi3
    # file_list = ['test_formatted_seen_query-seen_doc_test.jsonl','test_formatted_unseen_query-seen_doc_test.jsonl','test_formatted_seen_query-unseen_doc_test.jsonl','test_formatted_unseen_query-unseen_doc_test.jsonl']
    # MODEL_LIST =  ['prefix_only','title_url','title_url_doc','title_url_summary','title_url_yake','title_url_rag_dense','title_url_rag_sim_doc','title_url_rag_sparse']
    # op_file = os.path.join(DATA_PATH, "metrics", "eval_doclen_phi3_overall.csv")


    # L3
    # file_list = ['test_formatted_seen_query-seen_doc_test.jsonl','test_formatted_unseen_query-seen_doc_test.jsonl','test_formatted_seen_query-unseen_doc_test.jsonl','test_formatted_unseen_query-unseen_doc_test.jsonl']
    # MODEL_LIST =  ['prefix_only','title_url','title_url_doc','title_url_summary','title_url_yake','title_url_rag_dense','title_url_rag_sim_doc','title_url_rag_sparse']
    # op_file = os.path.join(DATA_PATH, "metrics", "eval_doclen_llama3_overall.csv")

    # GPT2
    # MODEL_LIST = ['no_doc','url','yake','summary','url_doc','rag_sparse_onedoc','rag_dense_simdoc','rag_dense_onedoc']
    # op_file = os.path.join(DATA_PATH, "metrics", "eval_gpt2_overall.csv")


    # T5
    # MODEL_LIST = ['no_doc','url','yake','summary','url_doc','rag_sparse_onedoc','rag_dense_simdoc','rag_dense_onedoc']
    # op_file = os.path.join(DATA_PATH, "metrics", "eval_t5_doclen_overall.csv")


    # T5 const docq
    # MODEL_LIST = ['t5const_docq-tries-rag_dense_onedoc','t5const_docq-tries-rag_dense_simdoc','t5const_docq-tries-rag_sparse_onedoc','t5const_docq-tries-summary','t5const_docq-tries-title_url_doc','t5const_docq-tries-title_url','t5const_docq-tries-yake','t5const_docq-tries-no_doc']
    # op_file = os.path.join(DATA_PATH, "metrics", "eval_t5const_docq_overall.csv")

    # T5 const global
    MODEL_LIST = ['t5const_global-tries-rag_dense_onedoc','t5const_global-tries-rag_dense_simdoc','t5const_global-tries-rag_sparse_onedoc','t5const_global-tries-summary','t5const_global-tries-title_url_doc','t5const_global-tries-title_url','t5const_global-tries-yake','t5const_global-tries-no_doc']
    op_file = os.path.join(DATA_PATH, "metrics", "eval_doclen_t5const_global_overall.csv")
    
    #op_file = os.path.join(DATA_PATH,"metrics","eval_doclen_t5_const_global_overall.csv")
    print(f"Output will be saved to: {op_file}")
    
    sb_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Initialize the final DataFrame for all results
    results_df = pd.DataFrame(columns=['model','doc_length_words','data_size','mrr','ndcg_pprec','ndcg_prec','bleu_rr','ndcg_alpha','sbert_mrr','tes'])

    # Read document length metadata once
    try:
        test_df = pd.read_csv("datasets/master/docs/trec_test.csv")
        print("Successfully loaded document length metadata.")
    except FileNotFoundError:
        print("Error: Document length file 'datasets/master/docs/trec_test.csv' not found. Exiting.")
        exit()

    # --- Model Processing Loop ---
    for model_name in MODEL_LIST:
        print(f"\n--- Processing Model: {model_name} ---")
        
        # MODIFICATION: Initialize collectors once per model to aggregate across all files.
        doc_len_buckets = ['0-200', '201-500', '501-1500', '1500+']
        metric_dict = {bucket: {
            'mrr_data_all':[], 'ndcg_partial_prec_all':[], 'ndcg_partial_rec_all':[],
            'ndcg_alpha_all':[], 'bleu_rr_all':[], 'sbert_data_all':[]
        } for bucket in doc_len_buckets}
        
        tes_dict = {bucket: [] for bucket in doc_len_buckets}
        all_model_data = [] # To store all rows for TES calculation later

        # --- File Aggregation Loop ---
        for file in file_list:
            print(f"  > Reading file: {file}")
            filepath = os.path.join(DATA_PATH, model_name, file)

            if not os.path.exists(filepath):
                print(f"    ... File not found, skipping.")
                continue

            # Read data from the current file
            data = []
            with open(filepath, encoding="utf-8") as f:
                for idx, row in enumerate(f, 1):
                    row = row.strip()
                    if row:
                        try:
                            data.append(json.loads(row))
                        except json.JSONDecodeError:
                            print(f"    ... Skipping invalid JSON at line {idx} in {file}")
            
            all_model_data.extend(data) # Aggregate data for the entire model
            print(f"    ... Loaded {len(data)} records.")

            # Process data for standard metrics
            for row in tqdm(data, desc=f"  > Processing {file}"):
                docid = row.get('docid') or row.get('doc_id')
                if docid is None:
                    continue

                try:
                    l_doc = test_df.loc[test_df['docid'] == docid, 'body_length'].iloc[0]
                except (IndexError, KeyError):
                    continue # Skip if docid not found in metadata

                gt = row['query']
                pred_list = row['completions']
                
                ndcg_pprec, ndcg_prec, mrr_data, ndcg_alpha_q, bleu_rr_q, sbert_data = sample_metric(sb_model, gt, pred_list, k=10)

                # Assign to the correct document length bucket
                bucket_key = ''
                if 1 <= l_doc <= 200: bucket_key = '0-200'
                elif 201 <= l_doc <= 500: bucket_key = '201-500'
                elif 501 <= l_doc <= 1500: bucket_key = '501-1500'
                elif l_doc > 1500: bucket_key = '1500+'
                else: continue
                
                metric_dict[bucket_key]['mrr_data_all'].append(mrr_data)
                metric_dict[bucket_key]['ndcg_partial_prec_all'].append(ndcg_pprec)
                metric_dict[bucket_key]['ndcg_partial_rec_all'].append(ndcg_prec)
                metric_dict[bucket_key]['ndcg_alpha_all'].append(ndcg_alpha_q)
                metric_dict[bucket_key]['bleu_rr_all'].append(bleu_rr_q)
                metric_dict[bucket_key]['sbert_data_all'].append(sbert_data)
        
        # --- Aggregated Calculation (Post-File Loop) ---
        print(f"\n--- Aggregating all metrics for model: {model_name} ---")
        
        # Calculate TES score from all aggregated data
        if all_model_data:
            df = pd.DataFrame(all_model_data)
            df['docid'] = df.apply(lambda row: row.get('docid') or row.get('doc_id'), axis=1)
            unique_pairs = df.drop_duplicates(subset=['query', 'docid'])
            
            for _, unique_row in tqdm(unique_pairs.iterrows(), total=len(unique_pairs), desc="  > Calculating TES"):
                query = unique_row['query']
                docid = unique_row['docid']
                
                try:
                    l_doc = test_df.loc[test_df['docid'] == docid, 'body_length'].iloc[0]
                except (IndexError, KeyError):
                    continue

                sub_df = df[(df['query'] == query) & (df['docid'] == docid)]
                pred_list = sub_df['completions'].tolist()
                tes_query = tes(query, pred_list)
                
                if 1 <= l_doc <= 200: tes_dict['0-200'].append(tes_query)
                elif 201 <= l_doc <= 500: tes_dict['201-500'].append(tes_query)
                elif 501 <= l_doc <= 1500: tes_dict['501-1500'].append(tes_query)
                elif l_doc > 1500: tes_dict['1500+'].append(tes_query)

        # Finalize and store results in a DataFrame
        metric_final = {}
        for bucket in doc_len_buckets:
            bucket_data = metric_dict[bucket]
            metric_final[bucket] = {
                'data_size': len(bucket_data['mrr_data_all']),
                'mrr': mean_reciprocal_rank(bucket_data['mrr_data_all']),
                'ndcg_pprec': calculate_average(bucket_data['ndcg_partial_prec_all']),
                'ndcg_prec': calculate_average(bucket_data['ndcg_partial_rec_all']),
                'bleu_rr': calculate_average(bucket_data['bleu_rr_all']),
                'ndcg_alpha': calculate_average(bucket_data['ndcg_alpha_all']),
                'sbert_mrr': mean_reciprocal_rank(bucket_data['sbert_data_all']),
                'tes': calculate_average(tes_dict[bucket])
            }
            
        metric_subdf = pd.DataFrame(metric_final).T.reset_index()
        metric_subdf.rename(columns={'index': 'doc_length_words'}, inplace=True)
        metric_subdf['model'] = model_name
        
        # Concatenate model's aggregated results to the main DataFrame
        results_df = pd.concat([results_df, metric_subdf], ignore_index=True)
        print(f"--- Aggregation complete for {model_name} ---")

    # --- Final Output ---
    print("\nAll models processed. Final aggregated results:")
    print(results_df)
    results_df.to_csv(op_file, index=False)
    print(f"\nResults successfully saved to {op_file}")