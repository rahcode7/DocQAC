import lucene
from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, TextField,StringField
from org.apache.lucene.index import IndexWriter, IndexWriterConfig, DirectoryReader, Term
from org.apache.lucene.search import IndexSearcher, PhraseQuery
from org.apache.lucene.store import FSDirectory
import shutil, os
import pandas as pd 
from tqdm import tqdm 

if not lucene.getVMEnv():
    lucene.initVM()

# Clear index directory if exists
index_path = "index"
if os.path.exists(index_path):
    shutil.rmtree(index_path)

# Setup index
directory = FSDirectory.open(Paths.get(index_path))
analyzer = StandardAnalyzer()
config = IndexWriterConfig(analyzer)
writer = IndexWriter(directory, config)

# df = pd.DataFrame([
#     {"doc_id": "1", "body": "this is a hello world example"},
#     {"doc_id": "2", "body": "this is a"},
#     {"doc_id": "3", "body": "welcome to the world of search"},
# ])


#Index all documents from DataFrame
DOC_PATH = r"datasets"
doc_df = pd.read_csv(os.path.join(DOC_PATH,"trec_orcas_docs.tsv"),sep="\t")
doc_df.columns = ['docid', 'url', 'title', 'body']
doc_df = doc_df.head(500) # Sample indexing
print(doc_df.columns)
print(doc_df.head(3))
print(doc_df.shape)

# Fill na and normalize document body
doc_df["body"] = doc_df["body"].fillna("").astype(str)
doc_df["body"] = doc_df["body"].str.lower()

# Store indexes
for _, row in tqdm(doc_df.iterrows(),total=doc_df.shape[0]):
    doc = Document()
    doc.add(StringField("docid", row["docid"], Field.Store.YES))  # Stored but not analyzed
    doc.add(TextField("content", row["body"], Field.Store.YES))     # Analyzed field
    writer.addDocument(doc)
writer.close()

