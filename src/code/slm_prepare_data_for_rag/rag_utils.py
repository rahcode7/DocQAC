from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_community.retrievers import BM25Retriever
from langchain_community.retrievers import BM25Retriever
#from langchain.retrievers import BM25Retriever
from langchain.docstore.document import Document
import pandas as pd 
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
import time 
from typing import List
import pickle

encoder = tiktoken.encoding_for_model("gpt-4")

### Dense retrieval

def rag_loader_dense(chunk_size,overlap):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    #embeddings = HuggingFaceEmbeddings(model_name="Snowflake/snowflake-arctic-embed-l-v2.0") # For 2nd version

    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,  # chunk size (characters)
            chunk_overlap=overlap,  # chunk overlap (characters)
            add_start_index=True,  # track index in original document
        )
    
    return embeddings,text_splitter

def get_chunks_dense(prefix,docid,text,embeddings,text_splitter,k=20):
    prefix = str(prefix)
    
    doc_vector_store = FAISS.load_local(
    f"datases/rag/vector_stores/all/{docid}", embeddings, allow_dangerous_deserialization=True 
    )
    
    #print("downloaded vector db index")
    doc = Document(page_content=text)
    all_splits = text_splitter.split_documents([doc])

    all_splits_docs = [doc.page_content for doc in all_splits]
    all_rel_chunks = []

    prefix_list = str(prefix).split()
    #top_c = int(k/len(prefix_list))
    top_c = k 
    sets = set()
    
    for prefix_word in prefix_list:
        #print(prefix_word)
        all_splits_filter = []
        rel_docs = []
        rel_chunks = []
        filtered_docs,sets  = filter_docs_by_prefix(prefix_word, all_splits_docs,sets)
        for item in filtered_docs:
            #print(item[0],item[1])
            all_splits_filter.append(all_splits[item[0]])

        ### Dense retriever
        #print(prefix_word,all_splits_filter)
        if all_splits_filter: 
            rel_docs = doc_vector_store.similarity_search(prefix,top_c)
            # list of chunks
            rel_chunks = [doc.page_content for doc in rel_docs]
            all_rel_chunks.append(rel_chunks)
        else:
            all_rel_chunks.append([])

    return all_rel_chunks


### Sparse retriever
def tokenize(text: str) -> List[str]:
    # Lowercase the input text
    lowered = text.lower()

    # Convert the lowered text into tokens
    tokens = encoder.encode(lowered)

    # Stringify the tokens 
    return [str(token) for token in tokens]

# def filter_docs_by_prefix(prefix, docs):
#     doc = [(i,doc) for i,doc in enumerate(docs) if any(word.lower().startswith(prefix) for word in doc.split())]
#     return (doc)

def filter_docs_by_prefix(prefix, docs, sets):
    #print(docs)
    # doc = [(i,doc) for i,doc in enumerate(docs) if any(word.lower().startswith(prefix) for word in doc.split())]
    #print(prefix,len(doc),doc[:top])

    doc_lst = []
    for i,doc in enumerate(docs):
        for word in doc.split():
            if word.lower().startswith(prefix):
                if doc not in sets:
                    sets.add(doc)
                    doc_lst.append((i,doc))
    return doc_lst, sets

def rag_sparse_loader():
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,  # chunk size (characters)
            chunk_overlap=30,  # chunk overlap (characters)
            add_start_index=True,  # track index in original document
        )
    
    return text_splitter

def get_chunks_sparse(prefix,text,text_splitter,k=30):
    prefix = str(prefix)
    #print(type(prefix))
    doc = Document(page_content=text)
    all_splits = text_splitter.split_documents([doc])

    all_splits_docs = [doc.page_content for doc in all_splits]
    all_rel_chunks = []

    
    prefix_list = str(prefix).split()
    #top_c = int(k/len(prefix_list))
    top_c = k 
    sets = set()

    for prefix_word in prefix_list:
        #print(prefix_word)
        all_splits_filter = []
        rel_docs = []
        rel_chunks = []
        filtered_docs,sets = filter_docs_by_prefix(prefix_word, all_splits_docs,sets)
        for item in filtered_docs:
            #print(item[0],item[1])
            all_splits_filter.append(all_splits[item[0]])

        #print(all_splits_filter)

        ### BM25 retriever
        #print(prefix_word,all_splits_filter)
        if all_splits_filter: 
            bm25_retriever = BM25Retriever.from_documents(all_splits_filter,k=top_c,preprocess_func=tokenize)
            rel_docs = bm25_retriever.get_relevant_documents(prefix)
            rel_chunks = [doc.page_content for doc in rel_docs]
            #print(prefix_word,rel_chunks)
            all_rel_chunks.append(rel_chunks)
        else:
            
            all_rel_chunks.append([])

        # If everything is empty
        if not all_rel_chunks:
            bm25_retriever = BM25Retriever.from_documents(all_splits_docs,k=top_c,preprocess_func=tokenize)
            bm25_retriever.get_relevant_documents(prefix)
            rel_chunks = [doc.page_content for doc in rel_docs]
            all_rel_chunks.append(rel_chunks)

    return all_rel_chunks

#### Similar Doc Dense retrieval

def similar_doc_loader(pickle_path):
    with open(pickle_path, "rb") as file:
        sim_doc_dict = pickle.load(file)
    return sim_doc_dict

def get_similar_docs(docid,sim_doc_dict):
    return sim_doc_dict[docid]

def get_chunks_similar_dense(prefix,docid,text,sim_docs,embeddings,text_splitter,k=40):
    prefix = str(prefix)
    all_rel_chunks = [] 

    top_c = int(k/len(sim_docs))
    #print("chunks per doc : ",top_c)

    # Get top_c dense matches for each of the similar docs
    for docid in sim_docs:

        doc_vector_store = FAISS.load_local(
        f"datasets/rag/vector_stores/all/{docid}", embeddings, allow_dangerous_deserialization=True 
        )
    
        rel_docs = doc_vector_store.similarity_search(prefix,top_c)
        rel_chunks = [doc.page_content for doc in rel_docs]
        #print("Document: ", docid,rel_chunks,len(rel_chunks))
        all_rel_chunks.append(rel_chunks)

    #print(len(all_rel_chunks))
    #print(len(all_rel_chunks[0]))
    return all_rel_chunks