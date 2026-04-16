from sentence_transformers import SentenceTransformer, models
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import os
import faiss

# === CONFIGURATION ===
SPLIT = "train"
print(f"Processing split: {SPLIT}")
base_dir = r"..\..\datasets\orcas"
op_dir = os.path.join(base_dir, "click-assign")
model_name = "microsoft/deberta-v3-base"

# === Load Model ===
word_embedding_model = models.Transformer(model_name, max_seq_length=512)
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True,
    pooling_mode_cls_token=False,
    pooling_mode_max_tokens=False
)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# === Load Data ===
if SPLIT == 'val':
    df = pd.read_csv(os.path.join(base_dir, "temporal_split", f"merged_query_url_4_days_{SPLIT}.tsv"),
                     delimiter='\t',
                     names=["qid", "query", "docid", "doc_url", "query_count", "doc_click_count"])
elif SPLIT == 'test':
    df = pd.read_csv(os.path.join(base_dir, "temporal_split", f"merged_query_url_10_days_{SPLIT}.tsv"),
                     delimiter='\t',
                     names=["qid", "query", "docid", "doc_url", "query_count", "doc_click_count"])
elif SPLIT == 'train':
    df = pd.read_csv(os.path.join(base_dir, "temporal_split", f"merged_query_url_1_month_{SPLIT}.tsv"),
                     delimiter='\t',
                     names=["qid", "query", "docid", "doc_url", "query_count", "doc_click_count"])
else:
    raise ValueError(f"No valid data found for split: {SPLIT}")

print("Sample raw data:")
print(df.head(3))

# === Load Query Embeddings ===
with open(os.path.join(op_dir, f"query2emb_{SPLIT}.pkl"), "rb") as f:
    query2emb = pickle.load(f)

query2freq = df.groupby("query")["query_count"].first().to_dict()
print("Loaded query embeddings")

# === Load FAISS Index ===
faiss_index_path = os.path.join(op_dir, f"faiss_query_index_{SPLIT}.faiss")
index = faiss.read_index(faiss_index_path)
print("Loaded FAISS index")

query_list = list(query2emb.keys())
query_mat = np.array([query2emb[q] for q in query_list])

# === Load Master DF ===
master_df = pd.read_csv(os.path.join(base_dir, "master", f"{SPLIT}_full.csv"), encoding="utf8")
master_sdf = master_df[master_df['query_type'] == 'similar']
master_cdf = master_df[master_df['query_type'] == 'clicked']
print("Dataset sizes:", master_df.shape, master_sdf.shape, master_cdf.shape)

# === Encode All Sim Queries ===
all_sim_queries = master_sdf['query'].unique().tolist()
sim_query2emb = dict(zip(
    all_sim_queries,
    model.encode(all_sim_queries, convert_to_numpy=True, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
))

# === Build docid → clicked queries mapping ===
docid2clicked_queries = defaultdict(list)
for _, row in df.iterrows():
    query = row['query']
    docid = row['docid']
    if query in query2emb:
        docid2clicked_queries[docid].append((query, row['query_count'], query2emb[query]))

# === Function: Estimate doc_click_count using only doc-local clicked queries ===
def estimate_doc_click_count(emb, docid, top_k=5):
    candidates = docid2clicked_queries.get(docid, [])
    if not candidates:
        return 0.0

    embs = np.array([x[2] for x in candidates])
    counts = np.array([x[1] for x in candidates])
    sims = np.dot(embs, emb.flatten())

    topk_idx = np.argsort(sims)[-top_k:]
    top_sims = sims[topk_idx]
    top_counts = counts[topk_idx]
    denom = np.sum(top_sims)
    return round(np.dot(top_sims, top_counts) / denom, 2) if denom > 0 else 0.0

# === Run Estimation Loop ===
result_list = []
for _, row in tqdm(master_sdf.iterrows(), total=master_sdf.shape[0], desc="Estimating doc-query click counts"):
    sim_query = row['query']
    clicked_query = row['clicked_query']
    docid = row['docid']

    emb = sim_query2emb[sim_query].astype("float32")
    est_clicks = estimate_doc_click_count(emb, docid, top_k=5)

    result_list.append({
        'clicked_query': clicked_query,
        'query': sim_query,
        'qid': row['qid'],
        'docid': docid,
        'doc_url': row['doc_url'],
        'query_count': 0,
        'doc_click_count': est_clicks,
        'query_length': row['query_length'],
        'query_type': 'similar',
        'id': row['id']
    })

# === Save Results ===
result_df = pd.DataFrame(result_list)
result_df.to_csv(os.path.join(op_dir, f"raw_click_est_doccnt_{SPLIT}.csv"), index=None)

master_df_final = pd.concat([result_df, master_cdf])
master_df_final.to_csv(os.path.join(op_dir, f"raw_{SPLIT}_doccnt_full.csv"), index=None)

print("Done.")
print("Saved result shapes:", result_df.shape, master_cdf.shape, master_df_final.shape)
