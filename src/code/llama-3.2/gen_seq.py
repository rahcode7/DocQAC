import pandas as pd
import torch
from transformers import AutoTokenizer # type: ignore
import argparse
import pickle
from tqdm import tqdm
import random

def get_complete_suffix(prefix, complete):
    space_index = prefix.rfind(" ")
    if (space_index == -1):
        partial = prefix
        suffix = complete
    else:
        partial = prefix[:space_index]
        suffix = complete[space_index+1:]
    return (partial, suffix)

def split_query(query, r_indx):
    # r_indx = random.randint(2, len(query)-1)

    prefix = query[:r_indx]
    _, complete_suffix = get_complete_suffix(prefix, query)
    return prefix, complete_suffix

# Initialize tokenizer globally
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tokenizer.pad_token = "<|reserved_special_token_0|>"  # use unk rather than eos token to prevent endless generation
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

def process_data(data_path, output_path, per_doc):
    """
    Loads data, processes it to create sequences, and saves them to a pickle file.
    """
    print(f"Loading tokenizer: t5-small")
    # Tokenizer is already loaded globally

    print(f"Loading data from: {data_path}")
    if ".tsv" in data_path:
        data = pd.read_csv(data_path, sep="\t")
    else:
        data = pd.read_csv(data_path)

    print("Processing data...")
    if per_doc:
        seq_dct = {}
    else:
        seq_lst = []
        
    for index, d_row in tqdm(data.iterrows(), total=data.shape[0], desc="Processing rows"):
        s = str(d_row.get("query", "")) # Ensure 'doc_url' exists and convert to string
        docid = str(d_row.get("docid", "")) # Ensure 'doc_id' exists and convert to string
        if not s: # Skip if doc_url is empty or not found
            print(f"Warning: Missing 'doc_url' or empty at row {index}. Skipping.")
            continue

        for r in range(1, len(s)):
            in_val, out_val = split_query(s, r)
            
            input_encoded = tokenizer(in_val, padding=False, add_special_tokens=False)
            output_encoded = tokenizer(out_val, padding=False, add_special_tokens=False)

            sequence = input_encoded['input_ids'] + [2**20] + output_encoded['input_ids'] + [tokenizer.eos_token_id]  # 2**20 is a placeholder for the separator token

            if per_doc:
                if docid not in seq_dct:
                    seq_dct[docid] = []
                seq_dct[docid].append(sequence)
            else:
                seq_lst.append(sequence)


    print(f"Saving processed sequences to: {output_path}")
    with open(output_path, 'wb') as f:
        if per_doc:
            pickle.dump(seq_dct, f)
        else:
            pickle.dump(seq_lst, f)

    print("Processing complete.")

if __name__ == "__main__":
    # python3 src/code/llama-3.2/gen_seq.py --data_path datasets/master/queries/train.csv --output_path src/code/llama-3.2/seq_list_doc_llama.pkl --per_doc
    parser = argparse.ArgumentParser(description="Process text data to create token sequences.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input data file (TSV or CSV).")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output sequence list (pickle file).")
    parser.add_argument("--per_doc", action='store_true')

    args = parser.parse_args()

    process_data(args.data_path, args.output_path, args.per_doc)