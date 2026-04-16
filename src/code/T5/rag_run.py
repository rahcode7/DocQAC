from rag_utils import rag_loader_dense,get_chunks_dense,rag_sparse_loader,get_chunks_sparse
from rag_utils import similar_doc_loader,get_similar_docs,get_chunks_similar_dense
import numpy as np 

if __name__ == "__main__":
    #docid = "D260827"
    #prefix = "arnold schwarzenegger dea"

    docid = "D1779712"
    prefix = "la care eligibility"
    #prefix = float(3.0)
    #prefix = np.nan
    #print(prefix)
    
    doc = "Search Now Language Assistance Member Sign in Provider Sign in IMPROVE YOUR MIND AND BODY WITH YOGA FOR SENIORSDon't miss out on free classes and services at our Inglewood Center!"

    # embeddings,text_splitter = rag_loader_dense(chunk_size=200,overlap=30)
    # rel_chunks = get_chunks_dense(prefix,docid,doc,embeddings,text_splitter,k=20)

    # 1. RAG sparse
    # text_splitter = rag_sparse_loader()
    # rel_chunks = get_chunks_sparse(prefix,doc,text_splitter)

    # 3. Similar docs dense retrieval
    pickle_path="datasets/rag/similar_docs/similar_docs.pkl"
    embeddings,text_splitter = rag_loader_dense(chunk_size=200,overlap=30)
    sim_doc_dict = similar_doc_loader(pickle_path)
    sim_doc_list = get_similar_docs(docid,sim_doc_dict)
    #print(docid,sim_doc_list)
    rel_chunks = get_chunks_similar_dense(prefix,docid,doc,sim_doc_list,embeddings,text_splitter,k=40)

    #print(rel_chunks)
