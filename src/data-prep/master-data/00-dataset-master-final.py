
import pandas as pd
import os
from tqdm import tqdm
import json
import random
tqdm.pandas()
pd.set_option('display.max_rows', None)
import sys


base_dir = r"datasets\orcas\temporal_split"
sim_path = r"datasets\orcas\fullmatch\fullmatched_nodup_filtered_all.jsonl" 
output_path = r"DocQAS\datasets\orcas\master"

# %%
with open(sim_path , "r", encoding="utf-8") as f:
    total_lines = sum(1 for _ in f)

rows = []
with open(sim_path , "r", encoding="utf-8") as infile:
    for line in tqdm(infile, total=total_lines, desc="Reading Q-SimQ"):
        data = json.loads(line)
        # query = next(iter(entry))
        # sim_queries = entry[query]
        
        for query, similar_queries in data.items():
            for similar_query in similar_queries:
                rows.append({'query': query, 'similar_query': similar_query})

df = pd.DataFrame(rows)
print(df.shape)
print(df.head(3))


  

# %%
df['query_len'] = df['query'].progress_apply(lambda x : len(x))
df['similar_query_len'] = df['similar_query'].progress_apply(lambda x : len(x))
print(df.head(3)) 
print(df.describe())

# %%
print(df.shape)
df = df[df['similar_query_len']>=3]
print(df.shape)

# %%
NUM_Q_SAMPLES = 750000

# %%
# Unique query pool
unique_q = df['query'].unique().tolist()
unique_simq = df['similar_query'].unique().tolist()
print(len(unique_q),len(unique_simq)) # 1225606 1912192
unique_allqueries = unique_q + unique_simq
print(len(set(unique_allqueries)))

# Sample 1 million

random.seed(123)
sampled_queries =  random.sample(unique_allqueries, k=NUM_Q_SAMPLES)
print(len(sampled_queries))
print(len(unique_allqueries))
print(len(sampled_queries))

#sampled_queries = unique_allqueries
print(len(sampled_queries))
print(len(set(sampled_queries)))

# %%
# Fetch queries related to similar queries 

# Common sampled similar queries
common_sampled_simq = set(sampled_queries).intersection(set(unique_simq))
common_sampled_uniqueq = set(sampled_queries).intersection(set(unique_q))
print(len(common_sampled_simq))
print(len(common_sampled_uniqueq))

# %% [markdown]
# ### Original train/test sets

# %%
# initial 1 month (4 weeks)
train_path = os.path.join(base_dir, "train.csv")
# next two weeks (temporally exclusive: 2 weeks)
test_path = os.path.join(base_dir, "test.csv")
val_path = os.path.join(base_dir, "val.csv")


train_df = pd.read_csv(train_path, delimiter=',')
print(len(train_df))

train_df['query_length'] = train_df['query'].progress_apply(lambda x : len(x))
train_df['query_length'].describe()

test_df = pd.read_csv(test_path, delimiter=',')

test_df['query_length'] = test_df['query'].progress_apply(lambda x : len(x))
test_df['query_length'].describe()

print(len(test_df))

val_df = pd.read_csv(val_path, delimiter=',')
val_df['query_length'] = val_df['query'].progress_apply(lambda x : len(x))
val_df['query_length'].describe()




# %%
print(train_df.shape,test_df.shape,val_df.shape)

# %%
train_df.head(3)

# %%
train_df['query'].nunique()

# %% [markdown]
# ### Merging
# Merge Unique queries (Both similar and actual query) with our train\test\dev sets

# %% [markdown]
# ### Similar Query Dataset 

# %%

### Get Similar Queries Docids and other columns 
sampled_sim_qdf = df[df['similar_query'].isin(common_sampled_simq)]
print(sampled_sim_qdf.shape)
print(sampled_sim_qdf.head(3))
sampled_sim_qdf.drop_duplicates(inplace=True)


sampled_sim_qdf = sampled_sim_qdf[['query','similar_query']]

# Training set 
# Do same for val and test set

print("TRAIN DATA SET SAMPLED ")
train_ssqdf = pd.merge(sampled_sim_qdf,train_df,on='query',how="inner")
#train_ssqdf.drop('query',axis=1,inplace=True)
train_ssqdf = train_ssqdf.rename(columns={'similar_query':'query','query':'clicked_query'})
train_ssqdf['query_type']='similar'
print(train_ssqdf.shape)
print(train_ssqdf['query'].nunique())
train_ssqdf.head(3)
# print(train_ssqdf[train_ssqdf['query'] ==train_ssqdf['clicked_query']])
# print(train_ssqdf[train_ssqdf['query'] ==train_ssqdf['clicked_query']].shape)

print("TEST DATA SET SAMPLED ")
test_ssqdf = pd.merge(sampled_sim_qdf,test_df,on='query',how="inner")
#train_ssqdf.drop('query',axis=1,inplace=True)
test_ssqdf = test_ssqdf.rename(columns={'similar_query':'query','query':'clicked_query'})
test_ssqdf['query_type']='similar'
print(test_ssqdf.shape)
print(test_ssqdf['query'].nunique())
print(test_ssqdf[test_ssqdf['query'] ==test_ssqdf['clicked_query']].shape)
test_ssqdf.head(3)

print("VAL DATA SET SAMPLED ")
val_ssqdf = pd.merge(sampled_sim_qdf,val_df,on='query',how="inner")
#train_ssqdf.drop('query',axis=1,inplace=True)
val_ssqdf = val_ssqdf.rename(columns={'similar_query':'query','query':'clicked_query'})
val_ssqdf['query_type']='similar'
print(val_ssqdf.shape)
print(val_ssqdf['query'].nunique())
val_ssqdf.head(3)



# %% [markdown]
# ### Clicked Query Dataset 

# %%

### Get Similar Queries Docids and other columns 

sampled_cl_qdf = pd.DataFrame(common_sampled_uniqueq,columns=['query'])
sampled_cl_qdf['similar_query'] = sampled_cl_qdf['query']
print(sampled_cl_qdf.shape)

# Training set 
# Do same for val and test set
print("TRAIN DATA SET SAMPLED ")
train_scqdf = pd.merge(sampled_cl_qdf,train_df,on='query')
#train_ssqdf.drop('query',axis=1,inplace=True)
train_scqdf = train_scqdf.rename(columns={'similar_query':'query','query':'clicked_query'})
train_scqdf['query_type']='clicked'
print(train_scqdf.shape)
print("Uniquer Queries ",train_scqdf['query'].nunique())
train_scqdf.head(3)
print("duplicates")
#print(train_scqdf[train_scqdf['query'] ==train_scqdf['clicked_query']].shape)

print("TEST DATA SET SAMPLED ")
test_scqdf = pd.merge(sampled_cl_qdf,test_df,on='query')
#train_ssqdf.drop('query',axis=1,inplace=True)
test_scqdf = test_scqdf.rename(columns={'similar_query':'query','query':'clicked_query'})
test_scqdf['query_type']='clicked'
print(test_scqdf.shape)
print("Uniquer Queries ",test_scqdf['query'].nunique())
test_scqdf.head(3)

print("VAL DATA SET SAMPLED ")
val_scqdf = pd.merge(sampled_cl_qdf,val_df,on='query')
#train_ssqdf.drop('query',axis=1,inplace=True)
val_scqdf = val_scqdf.rename(columns={'similar_query':'query','query':'clicked_query'})
val_scqdf['query_type']='clicked'
print(val_scqdf.shape)
print("Uniquer Queries ", val_scqdf['query'].nunique())
val_scqdf.head(3)



# %% [markdown]
# ### Merge Clicked and Similar Master dataset

# %%
train_sqdf = pd.concat([train_scqdf,train_ssqdf],ignore_index=True)
test_sqdf = pd.concat([test_scqdf,test_ssqdf],ignore_index=True)
val_sqdf = pd.concat([val_scqdf,val_ssqdf],ignore_index=True)
print(train_sqdf.shape,test_sqdf.shape,val_sqdf.shape)
train_sqdf.head(3)

# %%
train_sqdf['query_type'].value_counts()

# %%
print(train_sqdf[train_sqdf['query'] ==train_sqdf['clicked_query']].shape)

# %%
train_sqdf.tail(3)

# %%
train_sqdf.duplicated().sum()

# %%
print(train_sqdf['query_type'].value_counts())
print(test_sqdf['query_type'].value_counts())
print(val_sqdf['query_type'].value_counts())


# %%
train_sqdf['query_length'] = train_sqdf['clicked_query'].progress_apply(lambda x : len(x))
test_sqdf['query_length'] = test_sqdf['clicked_query'].progress_apply(lambda x : len(x))
val_sqdf['query_length'] = val_sqdf['clicked_query'].progress_apply(lambda x : len(x))


# %% [markdown]
# ### Filtering 
# First get TREC Documents 
#

# %%
DOC_PATH = r"datasets\msmarco"

trec_df = pd.read_csv(os.path.join(DOC_PATH,"msmarco-docs.tsv"),sep="\t")
trec_df.columns = ['docid', 'url', 'title', 'body']
print(trec_df.head(3))
### Remove null documents

trec_df.dropna(subset=['body'], inplace=True)
trec_docids = trec_df['docid'].unique().tolist()
print(f'Trec doc count {len(trec_docids)}')


# %% [markdown]
# ##### Filtering Criteria

# %%
def doc_clean(doc: str):
    """Cleans a doc
       returns : a clean version of doc"""
    return doc 

def query_clean(df):
    # remove non-ascii characters from each query
    df['query'] = df['query'].replace(r'[^\x00-\x7F]+', '', regex=True)
    
    # Trim whitespaces
    df['query'] = df['query'].str.strip()

    return df


def apply_filter(df, MIN_QUERIES=10, MAX_QUEIRES=500):

    stats_df = pd.DataFrame(columns=['filter','num_docs','num_queries','num_rows','avg_qlength'])


    print(f'BEFORE FILTERING - Unique docs {df.docid.nunique()} and Unique queries {df.qid.nunique()} and Unique rows {df.shape[0]}')

    stats_df  = pd.concat([stats_df, pd.DataFrame([['before filtering', df.docid.nunique(), df.qid.nunique(),df.shape[0],df['query_length'].mean()]], columns=['filter','num_docs','num_queries','num_rows','avg_qlength'])], ignore_index=True)

    df = df.drop_duplicates()
    print(f'FILTERING - DEDUPLICATION (All)- Unique docs {df.docid.nunique()} and Unique queries {df.qid.nunique()} and Unique rows {df.shape[0]}')
    stats_df  = pd.concat([stats_df, pd.DataFrame([['dedup overall', df.docid.nunique(), df.qid.nunique(),df.shape[0],df['query_length'].mean()]], columns=['filter','num_docs','num_queries','num_rows','avg_qlength'])], ignore_index=True)

    df = df.drop_duplicates(subset=['query','docid'])
    print(f'FILTERING - DEDUPLICATION (D,Q)- Unique docs {df.docid.nunique()} and Unique queries {df.qid.nunique()} and Unique rows {df.shape[0]}')

    stats_df  = pd.concat([stats_df, pd.DataFrame([['dedup query,doc pair', df.docid.nunique(), df.qid.nunique(),df.shape[0],df['query_length'].mean()]], columns=['filter','num_docs','num_queries','num_rows','avg_qlength'])], ignore_index=True)
    

    df = df.groupby('docid').filter(lambda x: (x['qid'].count()>MIN_QUERIES) & (x['qid'].count()<MAX_QUEIRES))
    print(f'FILTERING - MIN MAX QUERIES - Unique docs {df.docid.nunique()} and Unique queries {df.qid.nunique()} and Unique rows {df.shape[0]}')
    
    stats_df  = pd.concat([stats_df, pd.DataFrame([['min max query count', df.docid.nunique(), df.qid.nunique(),df.shape[0],df['query_length'].mean()]], columns=['filter','num_docs','num_queries','num_rows','avg_qlength'])], ignore_index=True)

    df = query_clean(df)
    # subset queries - number of character greater than 3
    df= df[(df['query'].str.len()>=3)]
    print(f'FILTERING QUERY LEN - Unique docs {df.docid.nunique()} and Unique queries {df.qid.nunique()} and Unique rows {df.shape[0]}')
    
    stats_df  = pd.concat([stats_df, pd.DataFrame([['query len', df.docid.nunique(), df.qid.nunique(),df.shape[0],df['query_length'].mean()]], columns=['filter','num_docs','num_queries','num_rows','avg_qlength'])], ignore_index=True)


    df = df[df['docid'].isin(trec_docids)]
    print(f'FILTERING - TREC DOCS - Unique docs {df.docid.nunique()} and Unique queries {df.qid.nunique()} and Unique rows {df.shape[0]}') 

    stats_df  = pd.concat([stats_df, pd.DataFrame([['merge TREC docs', df.docid.nunique(), df.qid.nunique(),df.shape[0],df['query_length'].mean()]], columns=['filter','num_docs','num_queries','num_rows','avg_qlength'])], ignore_index=True)

    print(stats_df)
    return df,stats_df

# %%
train_sqdf.head(3)

# %%
print(train_sqdf.shape)
train_sqdf[['query','docid']].drop_duplicates().shape

# %%
train_sqdf[train_sqdf.duplicated(subset=['query','docid'], keep=False)].sort_values(by='query', ascending=False).head(10)

# %%
filtered_train_df,stats_train_df = apply_filter(train_sqdf)
stats_train_df['dataset'] = 'train'
print("final_filtered_data", len(filtered_train_df))
filtered_train_df['query_type'].value_counts()


# %%
filtered_train_df.head(3)

# %%
filtered_test_df,stats_test_df = apply_filter(test_sqdf)
stats_test_df['dataset'] = 'test'
print("final_filtered_data", len(filtered_test_df))
filtered_train_df['query_type'].value_counts()

# %%
filtered_val_df,stats_val_df = apply_filter(val_sqdf)
stats_val_df['dataset'] = 'val'
print("final_filtered_val_data", len(filtered_val_df))
filtered_train_df['query_type'].value_counts()

# %%
filtered_val_df[filtered_val_df['query'] == 'zoo']
#filtered_val_df[filtered_val_df['query'] == '10 amendments']

# %%

filtered_train_df[['query','docid']].drop_duplicates().shape


# %%
# filtered_train_df[['query','docid']].drop_duplicates(inplace=True)
# filtered_train_df.shape

print(filtered_train_df.shape)
filtered_train_df = filtered_train_df.drop_duplicates(subset=['query', 'docid'])
print(filtered_train_df.shape)


print(filtered_test_df.shape)
filtered_test_df = filtered_test_df.drop_duplicates(subset=['query', 'docid'])
print(filtered_test_df.shape)

print(filtered_val_df.shape)
filtered_val_df = filtered_val_df.drop_duplicates(subset=['query', 'docid'])
print(filtered_val_df.shape)



# %%
# filtered_train_df.drop('index',inplace=True)
# filtered_test_df.drop('index',inplace=True)
# filtered_val_df.drop('index',inplace=True)


filtered_train_df = filtered_train_df.reset_index(drop=True)
filtered_test_df = filtered_test_df.reset_index(drop=True)
filtered_val_df = filtered_val_df.reset_index(drop=True)


# filtered_train_df = filtered_train_df.drop(columns=['index'])
# filtered_test_df = filtered_test_df.drop(columns=['index'])
# filtered_val_df = filtered_val_df.drop(columns=['index'])

filtered_train_df['id'] = filtered_train_df.index
filtered_val_df['id'] = filtered_val_df.index
filtered_test_df['id'] = filtered_test_df.index


filtered_train_df.to_csv(os.path.join(output_path,"filtered_train.csv"),index=None)
filtered_val_df.to_csv(os.path.join(output_path,"filtered_val.csv"),index=None)
filtered_test_df.to_csv(os.path.join(output_path,"filtered_test.csv"),index=None)
filtered_train_df.head(3)
print(filtered_train_df.shape)


# %%
filtered_train_df.shape

# %%
filtered_train_df[['query','docid']].drop_duplicates().shape

# %%
print(filtered_train_df.shape)
print(filtered_test_df.shape)
print(filtered_val_df.shape)


# %%
stats_all = [stats_train_df,stats_test_df,stats_val_df]
stats_df = pd.concat(stats_all)
print(stats_df)
stats_df.to_csv(os.path.join(output_path,"stats_df.csv"),index=None)

# %% [markdown]
# #### Relevance Classification - Evaluation

# %%

filtered_train_df = pd.read_csv(os.path.join(output_path,"filtered_train.csv"))
filtered_test_df = pd.read_csv(os.path.join(output_path,"filtered_test.csv"))
filtered_val_df = pd.read_csv(os.path.join(output_path,"filtered_val.csv"))
print(filtered_train_df.columns)


task_path = r"datasets\task\output"
true_train = pd.read_csv(os.path.join(task_path,"true_train.csv"))
true_train .drop('query_relevance', axis=1,inplace=True)

true_test = pd.read_csv(os.path.join(task_path,"true_test.csv"))
true_test.drop('query_relevance', axis=1,inplace=True)

true_val = pd.read_csv(os.path.join(task_path,"true_val.csv"))
true_val.drop('query_relevance', axis=1,inplace=True)

print(true_train.head(3))

# %%
print("task - Datasets info")
print(true_train.shape,true_test.shape,true_val.shape)
print("task - Unique docid query pairs ")
print(true_train[['docid','query']].nunique())
print(true_test[['docid','query']].nunique())
print(true_val[['docid','query']].nunique())
true_train = true_train[['docid','query']]
true_test = true_test[['docid','query']]
true_val = true_val[['docid','query']]

true_train.drop_duplicates(inplace=True)
true_test.drop_duplicates(inplace=True)
true_val.drop_duplicates(inplace=True)


print("task - Datasets info - dedup ")
print(true_train.shape,true_test.shape,true_val.shape)

# %%
true_train.head(3)

# %%
filtered_train_df.head(3)
print(filtered_train_df.shape)
unique_count = filtered_train_df[['query', 'docid']].drop_duplicates().shape[0]
print(f"Unique combinations: {unique_count}")

# %%
print(filtered_train_df.shape,filtered_test_df.shape,filtered_val_df.shape)

filtered_train_true_df = pd.merge(filtered_train_df,true_train,on=['query','docid'],how="inner")
filtered_test_true_df = pd.merge(filtered_test_df,true_test,on=['query','docid'],how="inner")
filtered_val_true_df = pd.merge(filtered_val_df,true_val,on=['query','docid'],how="inner")
print(filtered_train_true_df.shape)
print(filtered_test_true_df.shape)
print(filtered_val_true_df.shape)

# %%
print(filtered_train_true_df['docid'].nunique())

print(filtered_test_true_df['docid'].nunique())
print(filtered_val_true_df['docid'].nunique())


# %%
print(filtered_train_true_df['query'].nunique())

print(filtered_test_true_df['docid'].nunique())
print(filtered_val_true_df['docid'].nunique())

# %%

# %%

# %%
import random
# sampling 10K document from train
random.seed(123)
train_docids = filtered_train_true_df['docid'].unique().tolist()
#sampled_docs =  random.sample(train_docids, k=10000) # Require or not
sampled_docs = train_docids
print(len(sampled_docs))

train = filtered_train_true_df[filtered_train_true_df["docid"].isin(sampled_docs)]
print(f'FILTERING - SAMPLE DOCS - Unique docs {train.docid.nunique()} and Unique queries {train.qid.nunique()} and Unique rows {train.shape[0]}') 

seen_queries = train["query"].unique().tolist()


#train['query_length'] = train['query'].progress_apply(lambda x : len(x))
print(train['query_length'].mean())
train['query_length'].describe()

# %%
train['query_type'].value_counts()


# %%
pd.read_csv(os.path.join(output_path,filtered_test_true_df)

# %% [markdown]
# #### Test splits

# %%
import random
# sampling 10K document from train
test_docids = filtered_test_true_df['docid'].unique().tolist()
#test_sampled_docs =  random.sample(test_docids, k=10000)

test_sampled_docs = test_docids
print(len(test_sampled_docs))

test = filtered_test_true_df[filtered_test_true_df["docid"].isin(test_sampled_docs)]
print(f'FILTERING - SAMPLE DOCS - Unique docs {test.docid.nunique()} and Unique queries {test.qid.nunique()} and Unique rows {test.shape[0]}') 

test['query_length'] = test['query'].progress_apply(lambda x : len(x))
print(test['query_length'].mean())

test['query_length'].describe()

test_seen_queries = test["query"].unique().tolist()

# %%
print(test['docid'].nunique(),test['query'].nunique(),test.shape)

# %%
test['query_type'].value_counts()

# %%
seen_seen_test = test[(test["docid"].isin(sampled_docs)) & (test["query"].isin(seen_queries))]
print(f'FILTERING - SAMPLE DOCS - Unique docs {seen_seen_test.docid.nunique()} and Unique queries {seen_seen_test.qid.nunique()} and Unique rows {seen_seen_test.shape[0]}')
seen_unseen_test = test[(~test["docid"].isin(sampled_docs)) & (test["query"].isin(seen_queries))]
print(f'FILTERING - SAMPLE DOCS - Unique docs {seen_unseen_test.docid.nunique()} and Unique queries {seen_unseen_test.qid.nunique()} and Unique rows {seen_unseen_test.shape[0]}')
unseen_seen_test = test[(test["docid"].isin(sampled_docs)) & (~test["query"].isin(seen_queries))]
print(f'FILTERING - SAMPLE DOCS - Unique docs {unseen_seen_test.docid.nunique()} and Unique queries {unseen_seen_test.qid.nunique()} and Unique rows {unseen_seen_test.shape[0]}')
unseen_unseen_test = test[(~test["docid"].isin(sampled_docs)) & (~test["query"].isin(seen_queries))]
print(f'FILTERING - SAMPLE DOCS - Unique docs {unseen_unseen_test.docid.nunique()} and Unique queries {unseen_unseen_test.qid.nunique()} and Unique rows {unseen_unseen_test.shape[0]}')


seen_seen_test.to_csv(os.path.join(output_path, "seen_query-seen_doc_test_all.csv"), sep=",", index=False)
seen_unseen_test.to_csv(os.path.join(output_path, "seen_query-unseen_doc_test_all.csv"), sep=",", index=False)
unseen_seen_test.to_csv(os.path.join(output_path, "unseen_query-seen_doc_test_all.csv"), sep=",", index=False)
unseen_unseen_test.to_csv(os.path.join(output_path, "unseen_query-unseen_doc_test_all.csv"), sep=",", index=False)

# %% [markdown]
# ##### Sampled Test Set

# %%
##
TEST_SAMPLES = 3000
seen_seen_test_sample = seen_seen_test.sample(n=TEST_SAMPLES, random_state=1,replace=False)
seen_unseen_test_sample = seen_unseen_test.sample(n=TEST_SAMPLES, random_state=1,replace=False)
unseen_seen_test_sample = unseen_seen_test.sample(n=TEST_SAMPLES, random_state=1,replace=False)
unseen_unseen_test_sample = unseen_unseen_test.sample(n=TEST_SAMPLES,random_state=1,replace=False)
print(seen_seen_test_sample.shape)

# %%
seen_seen_test_sample['query_length'] = seen_seen_test_sample['query'].progress_apply(lambda x : len(x))
seen_unseen_test_sample['query_length'] = seen_unseen_test_sample['query'].progress_apply(lambda x : len(x))
unseen_seen_test_sample['query_length'] = unseen_seen_test_sample['query'].progress_apply(lambda x : len(x))
unseen_unseen_test_sample['query_length'] = unseen_unseen_test_sample['query'].progress_apply(lambda x : len(x))

seen_seen_test_sample.to_csv(os.path.join(output_path, "seen_query-seen_doc_test.csv"), sep=",", index=False)
seen_unseen_test_sample.to_csv(os.path.join(output_path, "seen_query-unseen_doc_test.csv"), sep=",", index=False)
unseen_seen_test_sample.to_csv(os.path.join(output_path, "unseen_query-seen_doc_test.csv"), sep=",", index=False)
unseen_unseen_test_sample.to_csv(os.path.join(output_path, "unseen_query-unseen_doc_test.csv"), sep=",", index=False)

# %%
test.to_csv(os.path.join(output_path, "test_full.csv"), sep=",", index=False)
train.to_csv(os.path.join(output_path, "train_full.csv"), sep=",", index=False)



# %% [markdown]
# #### Val Set 
# Random sampled from 10% of train set

# %%
# Old
# sub_df =  filtered_val_true_df[filtered_val_true_df['docid'].map(filtered_val_true_df['docid'].value_counts()) > 20]
# val_docids = sub_df['docid'].unique().tolist()
#len(val_docids)

# New 
val_docids = filtered_val_true_df['docid'].unique().tolist()
len(val_docids)

random.seed(123)
#sampled_docs =  random.sample(val_docids, k=1000)
sampled_docs = val_docids
print(len(sampled_docs))

val = filtered_val_true_df[filtered_val_true_df["docid"].isin(sampled_docs)]
print(f'FILTERING - SAMPLE DOCS - Unique docs {val.docid.nunique()} and Unique queries {val.qid.nunique()} and Unique rows {val.shape[0]}') 


import math
train_10pc_cnt = math.ceil(train.shape[0]*0.10)

#n=37785
print(train_10pc_cnt)

n = train_10pc_cnt
print("n",n)
print(val.shape)
val_sample = val.sample(n, replace=False,random_state=123)

print(val_sample.shape,val_sample.head(3)) 
#,val_sample['query'].nunique(),val.shape

#val_sample = val.sample(n=train_10pc_cnt, random_state=1,replace=False)
val_sample.to_csv(os.path.join(output_path, "val_full.csv"), sep=",", index=False)
val_sample.shape

# %%
val_sample['query_type'].value_counts()

# %%
stats_df_sample = pd.DataFrame(columns=['dataset','filter','num_docs','num_queries','num_rows','avg_qlength'])

df = seen_seen_test
stats_df_sample  = pd.concat([stats_df_sample, pd.DataFrame([['seen_seen_test_all','all', df.docid.nunique(), df.qid.nunique(),df.shape[0],df['query_length'].mean()]], columns=['dataset','filter','num_docs','num_queries','num_rows','avg_qlength'])], ignore_index=True)

df = seen_unseen_test
stats_df_sample  = pd.concat([stats_df_sample, pd.DataFrame([['seen_unseen_test_all','all', df.docid.nunique(), df.qid.nunique(),df.shape[0],df['query_length'].mean()]], columns=['dataset','filter','num_docs','num_queries','num_rows','avg_qlength'])], ignore_index=True)

df = unseen_seen_test
stats_df_sample  = pd.concat([stats_df_sample, pd.DataFrame([['seen_unseen_test_all','all', df.docid.nunique(), df.qid.nunique(),df.shape[0],df['query_length'].mean()]], columns=['dataset','filter','num_docs','num_queries','num_rows','avg_qlength'])], ignore_index=True)

df = unseen_unseen_test
stats_df_sample  = pd.concat([stats_df_sample, pd.DataFrame([['seen_unseen_test_all','all', df.docid.nunique(), df.qid.nunique(),df.shape[0],df['query_length'].mean()]], columns=['dataset','filter','num_docs','num_queries','num_rows','avg_qlength'])], ignore_index=True)

df = seen_seen_test_sample
stats_df_sample  = pd.concat([stats_df_sample, pd.DataFrame([['seen_seen_test','sample', df.docid.nunique(), df.qid.nunique(),df.shape[0],df['query_length'].mean()]], columns=['dataset','filter','num_docs','num_queries','num_rows','avg_qlength'])], ignore_index=True)

df = seen_unseen_test_sample
stats_df_sample  = pd.concat([stats_df_sample, pd.DataFrame([['seen_unseen_test','sample', df.docid.nunique(), df.qid.nunique(),df.shape[0],df['query_length'].mean()]], columns=['dataset','filter','num_docs','num_queries','num_rows','avg_qlength'])], ignore_index=True)

df = unseen_seen_test_sample
stats_df_sample  = pd.concat([stats_df_sample, pd.DataFrame([['seen_unseen_test','sample', df.docid.nunique(), df.qid.nunique(),df.shape[0],df['query_length'].mean()]], columns=['dataset','filter','num_docs','num_queries','num_rows','avg_qlength'])], ignore_index=True)

df = unseen_unseen_test_sample
stats_df_sample  = pd.concat([stats_df_sample, pd.DataFrame([['seen_unseen_test','sample', df.docid.nunique(), df.qid.nunique(),df.shape[0],df['query_length'].mean()]], columns=['dataset','filter','num_docs','num_queries','num_rows','avg_qlength'])], ignore_index=True)

df = train
stats_df_sample  = pd.concat([stats_df_sample, pd.DataFrame([['train','10000docs', df.docid.nunique(), df.qid.nunique(),df.shape[0],df['query_length'].mean()]], columns=['dataset','filter','num_docs','num_queries','num_rows','avg_qlength'])], ignore_index=True)

df = val_sample
stats_df_sample  = pd.concat([stats_df_sample, pd.DataFrame([['val','10000docs', df.docid.nunique(), df.qid.nunique(),df.shape[0],df['query_length'].mean()]], columns=['dataset','filter','num_docs','num_queries','num_rows','avg_qlength'])], ignore_index=True)

stats_df_sample.to_csv(os.path.join(output_path, "master_data_statistics.csv"), sep=",", index=False)



# %% [markdown]
# #### Get documents

# %%
train_10kdocs = train['docid'].unique()
print(len(train_10kdocs))
trec_train_df = trec_df[trec_df['docid'].isin(train_10kdocs)]
trec_train_df.head(3)

test_10kdocs = test['docid'].unique()
print(len(test_10kdocs))
trec_test_df = trec_df[trec_df['docid'].isin(test_10kdocs)]
trec_test_df.head(3)

val_docs = val_sample['docid'].unique()
print(len(val_docs))
trec_val_df = trec_df[trec_df['docid'].isin(val_docs)]
trec_val_df.head(3)


# %%
len(trec_train_df['body'].head(3).tolist()[2].split())

# %%
trec_train_df['body_length'] =trec_train_df['body'].progress_apply(lambda x : len(x.split()))
print(trec_train_df['body_length'].describe())

trec_test_df['body_length'] = trec_test_df['body'].progress_apply(lambda x : len(x.split()))
print(trec_test_df['body_length'].describe())

trec_val_df['body_length'] = trec_val_df['body'].progress_apply(lambda x : len(x.split()))
print(trec_val_df['body_length'].describe())


# %%
output_path

# %%
trec_train_df.to_csv(os.path.join(output_path, "trec_train.csv"),index=None)
trec_val_df.to_csv(os.path.join(output_path, "trec_val.csv"),index=None)
trec_test_df.to_csv(os.path.join(output_path, "trec_test.csv"),index=None)
