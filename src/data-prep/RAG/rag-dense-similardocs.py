from sentence_transformers import SentenceTransformer
import collections
import numpy as np
import pickle 
import pandas as pd 
import os 
 

if __name__ == "__main__":

    #model = SentenceTransformer("all-MiniLM-L6-v2")
    model = SentenceTransformer("sentence-transformers/msmarco-distilbert-base-v4") # Tuned for cosine similarity - as document length of different sizes   # dim 784

    top_k=10

    # Data Paths
    DATA_PATH = "datasets/master/" 
    OUTPUT_PATH = "datasets/rag/similar_docs/val"
    pickle_path = os.path.join(OUTPUT_PATH,"similar_docs.pkl")

    #Concatenate datasets
    df1 = pd.read_csv(os.path.join(DATA_PATH,"trec_train.csv"))
    df2 = pd.read_csv(os.path.join(DATA_PATH,"trec_test.csv"))
    df3 = pd.read_csv(os.path.join(DATA_PATH,"trec_val.csv"))
    df = pd.concat([df1,df2,df3],axis=0)
    print(df.shape)
    df.drop_duplicates(inplace=True)
    print(df.shape)

    #df = df.head(5)
    print(df.columns)
    
    docs = df.set_index("docid")["body"].to_dict()
    #print(docs)

    # Two lists of sentences
    #docs = {'d1':"The new movie is awesome",'d3':"The cat sits outside",'d2':"A man is playing guitar"}
    docs_dict = collections.OrderedDict(sorted(docs.items()))
    #print(docs_dict)
    docs_list = list(docs_dict.values())
    docs_keys = list(docs_dict.keys())
    #print(docs_keys)

    # # Compute embeddings for documents
    print("Creating embeddings of dim: ",model.encode("hello").shape[0])
    
    docs_embed = model.encode(docs_list)

    # Computer doc-doc similarity matrix
    print("Creating similarity matrix ..")
    sim_matrix = model.similarity(docs_embed,docs_embed).numpy()
    print("Generated similarity matrix")
    #print(sim_matrix)
    print(sim_matrix.shape)

    # get top-k matrix
    sim_idmatrix = np.argsort(sim_matrix, axis=1)[:, ::-1]
    #print(sim_idmatrix)
    topk_matrix = sim_idmatrix[:,:top_k]
    print(topk_matrix.shape)
    # print(topk_matrix)

    # Creat topk matrix of docids-docids
    dock_dict = collections.OrderedDict()
    rows,cols = topk_matrix.shape
    for i in range(rows):
        dock_dict[docs_keys[i]] = []
        for j in range(cols):
            #if topk_matrix[i][j] != i: # if doc itself is not included
            dock_dict[docs_keys[i]].append(docs_keys[topk_matrix[i][j]])

    with open(pickle_path, "wb") as file:
        pickle.dump(dock_dict, file)

    # Load the object
    with open(pickle_path, "rb") as file:
        dock_dict2 = pickle.load(file)

    print(len(dock_dict2))
