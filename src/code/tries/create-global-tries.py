import pandas as pd 
import numpy as np
from icecream import ic 
from tqdm import tqdm 
import random 
import argparse


def prefix_split(q: str,n: int):
    # Apply random split based base on number of prefixes
    """

    """
    l = len(q)  
    #i = np.random.randint(0,l//2)
    i = np.random.randint(0,l)
    #i_list = random.sample(xrange(0,l),n)
    #print(l,i,q[0:i],q)
    return q[0:i+1]
    # returns multiple query prefixes 

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    parser.add_argument('--input_file',type=str)
    args = parser.parse_args()


    # Read custom orcas dataset - the main one 
    # orcas_df = pd.read_csv(DATA_PATH + "orcas-qas-10k.csv")   

    orcas_df = pd.read_csv(args.input_file)

    #orcas_df.columns = map(str.lower, orcas_df.columns)
    print(orcas_df.columns)

    #orcas_df.columns = ['qid','query','docid','doc_url']
    ic(orcas_df.shape)
    ic(orcas_df['docid'].nunique())

    orcas_df.rename(columns={'query_count':'freq'},inplace=True)

    #sample = orcas_df[orcas_df.docid =='D1957503']
    #print(random.sample(range(1,500),100))

    # sample['freq'] = random.sample(range(1,500),sample.shape[0])
    # sample
    #sample[['query','freq']].to_txt()
    np.savetxt(r'datasets/inputs/all-queries-global.txt', orcas_df[['query','freq']].values, fmt='%s',delimiter='\t')
    print("dataset created : all-queries-global.txt")

    


