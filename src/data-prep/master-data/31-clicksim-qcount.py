from sentence_transformers import SentenceTransformer, models
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import os
import faiss

# === CONFIGURATION ===
SPLIT = "test"
print(SPLIT)
base_dir = r"..\..\datasets\orcas"
op_dir = os.path.join(base_dir, "click-assign")

# === Load Model ===
model_name = "microsoft/deberta-v3-base"
word_embedding_model = models.Transformer(model_name, max_seq_length=512)#, tokenizer_args={"use_fast": False})
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True,
    pooling_mode_cls_token=False,
    pooling_mode_max_tokens=False
)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# === Load Data ===
# df = pd.read_csv(os.path.join(base_dir, "temporal_split", f"{SPLIT}.csv"))
# print(df.shape)

# Or read raw counts
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
     print("no file found of raw counts")
print(df.head(3))

#  ===========================
print("Encoding unique queries...")
# unique_queries = df['query'].unique().tolist()
# query_embs = model.encode(unique_queries, convert_to_numpy=True,
#                           normalize_embeddings=True, show_progress_bar=True, batch_size=64)
# query2emb = dict(zip(unique_queries, query_embs))

# pickle_path = os.path.join(op_dir, f"query2emb_{SPLIT}.pkl")
# with open(pickle_path, "wb") as f:
#     pickle.dump(query2emb, f)

# === Load embeddings 

with open(os.path.join(op_dir, f"query2emb_{SPLIT}.pkl"), "rb") as f:
    query2emb = pickle.load(f)

query2freq = df.groupby("query")["query_count"].first().to_dict()
print("loaded q embeddings")

# === Build Index === 
# query_list = list(query2emb.keys())
# query_embs = np.array([query2emb[q] for q in query_list]).astype("float32")
# faiss.normalize_L2(query_embs)

# index = faiss.IndexFlatIP(query_embs.shape[1])
# print("Adding to index")
# index.add(query_embs)
# faiss_index_path = os.path.join(op_dir, f"faiss_query_index_{SPLIT}.faiss")
# faiss.write_index(index, faiss_index_path)
# === Load FAISS Index ===

faiss_index_path = os.path.join(op_dir, f"faiss_query_index_{SPLIT}.faiss")
index = faiss.read_index(faiss_index_path)
print("loaded index")

query_list = list(query2emb.keys())
query_mat = np.array([query2emb[q] for q in query_list])

# === Precompute embeddings for all sim queries ===
master_df = pd.read_csv(os.path.join(base_dir, "master", f"{SPLIT}_full.csv"), encoding="utf8")
master_sdf = master_df[master_df['query_type'] == 'similar']
master_cdf = master_df[master_df['query_type'] == 'clicked']
print("dataset actual sizes")
print(master_df.shape,master_cdf.shape,master_sdf.shape)

print("Encoding all sim queries...")
all_sim_queries = master_sdf['query'].unique().tolist()
print("Similar queries size",len(all_sim_queries))

sim_query2emb = dict(zip(
    all_sim_queries,
    model.encode(all_sim_queries, convert_to_numpy=True, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
))

# === Build docid mappings ===
docid2embs = defaultdict(list)
docid2clicks = defaultdict(list)
for _, row in df.iterrows():
    docid2embs[row['docid']].append(query2emb[row['query']])
    docid2clicks[row['docid']].append(row['query_count'])

query_index = defaultdict(list)
for _, row in df.iterrows():
    query_index[row['query']].append((row['docid'], row['query_count'], row['doc_click_count']))

# === Estimation Function ===
def estimate_query_count_with_faiss_emb(emb, top_k=5):
    _, topk_indices = index.search(emb.reshape(1, -1), top_k)
    topk_indices = topk_indices[0]
    
    sims = np.dot(query_mat[topk_indices], emb.flatten())  # since all embeddings are L2-normalized
    freqs = np.array([query2freq.get(query_list[i], 0) for i in topk_indices])
    denom = np.sum(sims)
    return round(np.dot(sims, freqs) / denom, 2) if denom != 0 else 0.0

# === Run Estimation ===
result_list = []
cnt = 0 
qid = 0 
for _, row in tqdm(master_sdf.iterrows(), total=master_sdf.shape[0], desc="Estimating clicks"):
    query = row['clicked_query']
    sim_query = row['query']
    docid = row['docid']
    emb = sim_query2emb[sim_query].astype("float32")
    est = estimate_query_count_with_faiss_emb(emb, top_k=5)

    result_list.append({
        'clicked_query': query,
        'query': sim_query,
        'qid': row['qid'],
        'docid': docid,
        'doc_url': row['doc_url'],
        'query_count': est,
        'doc_click_count': 0,
        'query_length': row['query_length'],
        'query_type': 'similar',
        'id': row['id']
    })
print(cnt)
print(qid)
result_df = pd.DataFrame(result_list)
result_df.to_csv(os.path.join(op_dir, f"raw_click_est_qcnt_{SPLIT}.csv"), index=None)

master_df = pd.concat([result_df, master_cdf])
master_df.to_csv(os.path.join(op_dir, f"raw_{SPLIT}_qcnt_full.csv"), index=None)
print(result_df.shape, master_sdf.shape, master_df.shape, master_cdf.shape)
