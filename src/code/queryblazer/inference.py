from queryblazer import *
import argparse
import json
import pickle
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description ='')
    parser.add_argument('file',type=str)
    parser.add_argument('opfile',type=str)
    args=parser.parse_args()

    config = Config(branch_factor=30, beam_size=100, topk=100, length_limit=100)
    qbz = QueryBlazer(encoder="encoder.fst", model="ngram.fst", config=config)
    qbz.LoadPrecomputed('precomputed.bin')
    
    # Load dataset for predictions
    results = []
    with open(args.file) as f:
        for query in f:
            completions = qbz.Complete(query)[0]
            results.append(completions)
    print("num preds ",len(results))
    with open(args.opfile,'wb') as fp:
        pickle.dump(results, fp)
    with open (args.opfile,'rb') as fp:
        itemlist = pickle.load(fp)
    print(itemlist[0:2])
