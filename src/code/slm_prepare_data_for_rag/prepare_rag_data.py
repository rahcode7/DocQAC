import argparse
import torch
import torch.multiprocessing as mp
from collections import defaultdict
from copy import deepcopy
import os
from transformers import (
    AutoTokenizer
    )
from tqdm import tqdm
import re
import math
from datetime import datetime
import pandas as pd

from rag_utils import rag_loader_dense,get_chunks_dense,rag_sparse_loader,get_chunks_sparse
from rag_utils import similar_doc_loader, get_similar_docs, get_chunks_similar_dense

import sys

def load_text_stream(load_path, max_limit=-1):
    index = 0
    with open(load_path, 'r', encoding='utf-8') as f:
        for row in f:
            index+=1
            input_text = row
            if input_text[-1]=='\n':
                input_text = input_text[:-1]
            if (max_limit>0 and index>max_limit):
                yield False, ""
            yield True, input_text
    yield False, ""

def prepare_for_rag(raw_doc_path, context_type):
    rag_embedding = None
    rag_text_splitter = None
    rag_raw_doc = None
    rag_sim_doc_dict = None

    if not os.path.exists(raw_doc_path):
        return rag_embedding, rag_text_splitter, rag_raw_doc

    if context_type == "rag_sparse":
        rag_text_splitter = rag_sparse_loader()
    elif context_type == "rag_dense":
        rag_embedding, rag_text_splitter = rag_loader_dense(chunk_size=200, overlap=30)
    elif context_type == "rag_sim_doc":
        #pickle_path="./Document-AS/rag/similar_docs/similar_docs.pkl"
        pickle_path="datasets/rag/similar_docs/similar_docs.pkl"
        rag_embedding, rag_text_splitter = rag_loader_dense(chunk_size=200,overlap=30)
        rag_sim_doc_dict = similar_doc_loader(pickle_path)

    train_data = pd.read_csv(os.path.join(raw_doc_path, "trec_train.csv"))[['docid', 'body']]
    val_data = pd.read_csv(os.path.join(raw_doc_path, "trec_val.csv"))[['docid', 'body']]
    test_data = pd.read_csv(os.path.join(raw_doc_path, "trec_test.csv"))[['docid', 'body']]
    # merge train and val data
    all_data = pd.concat([train_data, val_data, test_data])
    all_data = all_data.drop_duplicates()

    rag_raw_doc = {}
    for _, row in all_data.iterrows():
        rag_raw_doc[row['docid']] = row['body']
        
    return rag_embedding, rag_text_splitter, rag_raw_doc, rag_sim_doc_dict

# modifiy the load_text_stream to function to have begin and end indexes
def load_text_stream_with_index(load_path, max_limit=-1, start_index:int=0):
    index = -1
    with open(load_path, 'r', encoding='utf-8') as f:
        for row in f:
            index+=1
            if index<start_index:
                continue
            input_text = row
            if input_text[-1]=='\n':
                input_text = input_text[:-1]
            if (max_limit>0 and index>=max_limit):
                yield False, "", index
            yield True, input_text, index
    yield False, "", index

def normalize_text(text):
    text = text.lower().strip()
    text = re.sub('\s+', ' ', text)
    return text

def truncate_string(input_string, max_words):
    max_avg_chars = 5*max_words # space + 4 char word = 200 chars
    input_words = input_string.split(' ')[-max_words:]
    reconstructed_input = " ".join(input_words)
    return reconstructed_input[-max_avg_chars:]

def process_text(previous_query, current_query):
    combined_query = f"prev_query: {previous_query}, query: {current_query}"
    return combined_query

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)        

def extract_header(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        curr_line = f.readline()
        header = [tz.strip() for tz in curr_line.split('\t')]
    print(f"[header_info] column_count: {len(header)}")
    print(f'[header info] column_names: {" | ".join(header)}')
    return header

def convert_tsv_to_json(row, header):
    temp = {}
    for k,v in zip(header, row):
        temp[k] = v
    return temp

def add_rag_content_col(input_dict, rag_output, context_type):
    input_dict[context_type] = rag_output

def convert_json_to_tsv(input_json, header, context_type):
    final_str=""
    for h in header:
        final_str+=f"{input_json[h]}\t"
    # finally add the "similarity_score"
    final_str+=f"{input_json[context_type]}"
    return final_str

def get_prefix_count(header, input_file, max_limit, start_index):
    # iterate over the input file and count the number of lines
    count=0
    stream = load_text_stream_with_index(input_file, max_limit, start_index)
    for status, curr_input, index in tqdm(stream):
        if status==False:
            break
        temp = curr_input.split('\t')
        assert len(temp)==len(header), f"{len(temp)} != {len(header)} at line {index}"
        count+=1
    print("[ok] all instances in file are valid record")
    return count

# function to delete the file if it exists
def delete_file_if_exists(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

def truncate_string(input_string, max_words, side='left'):
    max_avg_chars = 5*max_words # space + 4 char word = 200 chars
    input_words = input_string.split(' ')[-max_words:]
    reconstructed_input = " ".join(input_words)
    if side == 'right':
        return reconstructed_input[:max_avg_chars]
    return reconstructed_input[-max_avg_chars:]

def tokenizer_budget(tokenizer, input_str, token_budget, truncation_side='left'):
    input_str = truncate_string(input_str, token_budget, side=truncation_side)
    encoded_tokens = tokenizer.encode(input_str, add_special_tokens=False)
    if len(encoded_tokens) > token_budget:
        encoded_tokens = encoded_tokens[-token_budget:]
    return tokenizer.decode(encoded_tokens)

def is_empty_string(input_str):
    if input_str is None:
        return True
    input_str = input_str.strip()
    return input_str == "" or len(input_str)==0

def process_chunks(tokenizer, wordwise_chunk_list, token_budget, top_k=20):
    # implement reciprocal rank fusion
    chuck_priority = defaultdict(lambda: 0.0)
    for word_level_chunks in wordwise_chunk_list:
        for idx, chunk in enumerate(word_level_chunks):
            if is_empty_string(chunk):
                continue
            chuck_priority[chunk] = chuck_priority[chunk] + 1/(idx+1)
    # sort by fused reciprocal rank
    sorted_chunks = sorted(chuck_priority.items(), key=lambda x: x[1], reverse=True)
    # return only top-k chunks based in ranks
    candidate_chunks = [chunk[0] for chunk in sorted_chunks[:top_k]]
    # truncate the chunks to fit the token budget
    if len(candidate_chunks) == 0:
        return "" 
    # fit the token budget to each chunk
    individual_token_budget = int(token_budget/len(candidate_chunks))
    processed_chunks = []
    for chunk in candidate_chunks:
        processed_chunks.append(tokenizer_budget(tokenizer, chunk, individual_token_budget, truncation_side='right'))

    return " | ".join(processed_chunks)

def prepare_data(args):
    # prepare the RAG utils
    rag_embedding, rag_text_splitter, rag_raw_doc, rag_sim_doc_dict = prepare_for_rag(args.raw_doc_file, args.context_type)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct", use_fast=True)
    print("[worker-%d] using device : %s" % (args.worker_id, args.device))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    output_file = open(args.output_file, 'w', encoding='utf-8')
    # special case handling (0th worker is only worker then write the header)
    if args.num_workers==1:
        # write header
        output_file.write("\t".join(args.header)+'\t'+args.context_type+'\n')
    prefix_count = args.prefix_count
    print("[worker-%d] records for processing : %d" % (args.worker_id, prefix_count))
    print("[worker-%d] records start index : %d, end_index : %d" % (args.worker_id, args.start_index, args.max_limit))
    args.inference_count = 0
    args.trimmed_prefixes = 0    
    with torch.no_grad():
        print("[worker-%d] starting the data processing ..." % args.worker_id)
        stream = load_text_stream_with_index(args.input_file, args.max_limit, args.start_index)
        with tqdm(total=prefix_count, desc="processing", unit=" lines", position=0, leave=True) as pbar:
            for status, record, _ in stream: 
                if status==False:
                    break
                curr_row = record.split('\t')
                curr_dict = convert_tsv_to_json(curr_row, args.header)
                prefix = curr_dict["prefix"]
                doc_id = curr_dict["docid"]

                if args.context_type == "rag_sparse":
                    chunks = get_chunks_sparse(prefix, rag_raw_doc.get(doc_id, ""), rag_text_splitter)
                elif args.context_type == "rag_dense":
                    chunks = get_chunks_dense(prefix, doc_id, rag_raw_doc.get(doc_id, ""), rag_embedding, rag_text_splitter)
                else:
                    target_sim_doc_list = get_similar_docs(doc_id, rag_sim_doc_dict)
                    chunks = get_chunks_similar_dense(prefix, doc_id, rag_raw_doc.get(doc_id, ""), target_sim_doc_list, rag_embedding, rag_text_splitter, k=40)
                
                processed_chunks = process_chunks(tokenizer, chunks, args.token_budget)
                add_rag_content_col(curr_dict, processed_chunks, args.context_type)
                output_file.write(convert_json_to_tsv(curr_dict, args.header, args.context_type)+'\n')
                args.inference_count+=1
                torch.cuda.empty_cache()
                pbar.update(1)
        
        output_file.close()
        print("[worker-%d] processing done : %d " % (args.worker_id, args.inference_count))

# function to print time delta in human readable format hh:mm:ss
def get_time_delta(time_delta):
    hours, rem = divmod(time_delta.seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)

def create_worker_file_name(idx, file_path):
    file_name = os.path.basename(file_path)
    parent_dir = os.path.dirname(file_path)
    os.makedirs(parent_dir, exist_ok=True)
    return os.path.join(parent_dir, f"worker-{idx}-{file_name}")

if __name__ == "__main__":

    exp_variations = [
        "rag_sparse",
        "rag_dense",
        "rag_sim_doc",
    ]

    parser = argparse.ArgumentParser()
    # Global model configuration
    parser.add_argument('--raw_doc_file', type=str, required=True,
                        help='')
    parser.add_argument('--input_file', type=str, required=True,
                        help='')
    parser.add_argument('--output_file', type=str, required=True,
                        help='')
    parser.add_argument('--context_type', type=str, choices=exp_variations, required=True,
                        help='')
    parser.add_argument('--max_limit', type=int, default=-1,
                        help='')
    parser.add_argument('--token_budget', type=int, default=300,
                        help='')
    # if multiple gpus devices are available
    parser.add_argument('--num_workers', type=int, default=1, 
                        help='total number of workers')                                     
    args = parser.parse_args() 

    global_start_time = datetime.utcnow()
    print("[global] argument passed :")
    for arg in vars(args):
        print("[global] %s - %s" % (arg, getattr(args, arg)))
    print('--'*30)
    args.header = extract_header(args.input_file)
    args.global_prefix_count = get_prefix_count(args.header, args.input_file, args.max_limit, 0)
    print("[global] total prefix count : %d" % args.global_prefix_count)

    if args.num_workers==1:
        args.worker_id = 0
        # skip the header
        args.start_index = 1
        args.prefix_count = args.global_prefix_count
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        prepare_data(args)
    else:
        mp.set_start_method('spawn', force=True)
        # Multi-GPU inference
        bucket_size = math.ceil(args.global_prefix_count/args.num_workers)
        worker_configs = []
        # creating the config
        for idx in range(args.num_workers):
            worker_config = deepcopy(args)
            worker_config.worker_id = idx
            # skip header
            worker_config.start_index = 1 if idx*bucket_size == 0 else idx*bucket_size 
            worker_config.max_limit = min(worker_config.start_index + bucket_size, args.global_prefix_count)
            worker_config.prefix_count = (worker_config.max_limit - worker_config.start_index)
            worker_config.device = f"cuda:{idx}" if torch.cuda.is_available() else "cpu"
            worker_config.output_file = create_worker_file_name(idx, os.path.abspath(args.output_file))
            worker_configs.append(worker_config)
        # spawning the process
        worker_processes = []
        print("[global] spawning %d process for processing the data" % args.num_workers)
        for idx in range(args.num_workers):
            p = mp.Process(target=prepare_data, args=(worker_configs[idx],))
            p.start()
            worker_processes.append(p)
        # wait for all the process to finish
        for p in worker_processes:
            p.join()
        
        # merge intermediate outputs to final file 
        final_output_file = open(args.output_file, 'w', encoding='utf-8')
        # write header
        final_output_file.write("\t".join(args.header)+'\t'+args.context_type+'\n')
        for wc in worker_configs:
            stream = load_text_stream(wc.output_file)
            for status, data in stream:
                if status==False:
                    break
                final_output_file.write("%s\n" % data)
        final_output_file.close()
        
        # delete the intermediate files 
        for wc in worker_configs:
            delete_file_if_exists(wc.output_file)

    time_delta = (datetime.utcnow() - global_start_time)
    print("[global] completed %d data processing in %s" % (args.global_prefix_count, get_time_delta(time_delta)))