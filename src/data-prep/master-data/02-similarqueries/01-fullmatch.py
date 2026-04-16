import lucene
from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, TextField,StringField
from org.apache.lucene.index import IndexWriter, IndexWriterConfig, DirectoryReader, Term
from org.apache.lucene.search import IndexSearcher, PhraseQuery
from org.apache.lucene.store import FSDirectory
import shutil, os
import pandas as pd
import json  
import re 
from tqdm import tqdm 

def clean_query(query):
    # Match StandardAnalyzer-like normalization
    return re.sub(r"[^\w]+", " ", query).strip().lower()

# Remove index lock if already used
lock_path = "index/write.lock"
if os.path.exists(lock_path):
    os.remove(lock_path)


if not lucene.getVMEnv():
    lucene.initVM()

# Setup index
index_path = "index"
directory = FSDirectory.open(Paths.get(index_path))
# Search setup
reader = DirectoryReader.open(directory)
searcher = IndexSearcher(reader)

### Test sample query

# "Highest Quality Safety Solutions"
# "›Geography Is"

# PhraseQuery: "hello world"
# builder = PhraseQuery.Builder()
# builder.add(Term("content", "highest"), 0)
# builder.add(Term("content", "quality"), 1)
# phrase_query = builder.build()

# # Execute search
# hits = searcher.search(phrase_query, 1).scoreDocs
# for hit in hits:
#     found_doc = searcher.doc(hit.doc)
#     print("Match:", found_doc.get("content"))
#     print("Match:", found_doc.get("docid"))
# reader.close()


### Full set of queries 

# Read input file and check similar queries
input_file = "datasets/merged_similarq_top100.jsonl"
output_path = "datasets/merged_similarq_top100_fullmatched.jsonl"

def is_phrase_in_index(phrase):
    """Check if a phrase exists in any document."""
    tokens = phrase.strip().split()
    if not tokens:
        return 0

    builder = PhraseQuery.Builder()
    for i, token in enumerate(tokens):
        builder.add(Term("content", token), i)
    phrase_query = builder.build()

    #hits = searcher.search(phrase_query, 1).scoreDocs
    #return 1 if hits else 0

    return 1 if searcher.count(phrase_query) > 0 else 0

cnt=0
total_lines = sum(1 for _ in open(input_file, "r", encoding="utf-8"))
with open(input_file, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
    for line in tqdm(infile, total=total_lines, desc="Processing queries"):
        cnt+=1
        if cnt>2:
            break
        if not line.strip():
            continue
        obj = json.loads(line)
        updated_similars = []
        for query, similars in obj.items():
            for phrase, rank in similars:
                phrase = clean_query(phrase)
                flag = is_phrase_in_index(phrase)
                updated_similars.append([phrase, rank, flag])
            obj[query] = updated_similars
        outfile.write(json.dumps(obj, ensure_ascii=False) + "\n")

reader.close()