### Master dataset preparation 
```
cd master-data

```
### Step1 
Subset Queries 
- from certain time period from Bing logs
- Overlap with ORCAS queries


### Step 2 Find similar queries

```
cd 02-similarqueries
```

###### Build Document Indexes 
1.Get MSMARCO documents from the train,test,dev set 

```
 python3 00-index-docs.py
```

###### Fetch queries in documents
3. Run full match
Single Core
```
 python3 01-fullmatch.py
```

### Step 3 Click assignment
# Assign doc click count

python 30-clicksim-dqcount.py

# Assing query count 
python 31-clicksim-dqcount.py




#### Input representation data preparation guide
1. YAKE (Keyphrase Extraction)
2. RAG 
3. Summaries
