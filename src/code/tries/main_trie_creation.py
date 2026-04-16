from tqdm import tqdm
import pickle
import argparse
from utils import QueryCompletion, preprocess, is_empty, load_text_stream, check_and_create_path, get_line_count
import sys
sys.setrecursionlimit(1000)

def is_empty(input_text):
    input_text = input_text.strip()
    if input_text=='' or len(input_text)==0 or len(input_text.split('\t'))<=1:
        return True
    return False


def init(args):
    query_completion = QueryCompletion()
    index = 0
    total_count = get_line_count(args.input_file, args.data_limit)
    print("creating the main trie using %d queries" % (total_count))
    with tqdm(total=total_count) as pbar:
        for is_valid, input_text in load_text_stream(args.input_file, args.data_limit):
            pbar.update(1)
            if not is_valid:
                break
            if is_empty(input_text): # or len(input_text)>50:
                continue
            query, frequency = input_text.split('\t')
            print(query)
            if float(frequency)<args.threshold:
                continue
            query_completion.insert(preprocess(query), float(frequency))
            print(index)
            index+=1
    print("Total queries: ", index)
    print("saving the main trie")
    with open(args.output_trie, 'wb') as f:
        pickle.dump(query_completion, f)
    print("main trie saved at: ", args.output_trie)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True,
                        help='file containing the query, frequency pairs.')
    parser.add_argument('--output_trie', type=str, required=True,
                        help='path to location to save the trie object.')
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument('--data-limit',
                        type=int,
                        default=-1,
                        help='maximum number of instances to load. -1 to include all.')
    args = parser.parse_args()
    check_and_create_path(args.output_trie)
    init(args)
