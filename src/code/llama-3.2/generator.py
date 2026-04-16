import argparse
import torch
import torch.multiprocessing as mp
from copy import deepcopy
import os
import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
    )
from tqdm import tqdm
import re
import math
from datetime import datetime
import json
from peft import PeftModel
from task_utils import prepare_input, TokenPrefixSuffixTrie, TrieConstrainedLogitsProcessor
import pickle


exp_variations = [
    "prefix_only",
    "title_doc",
    "title_url",
    "title_url_doc",
    "title_url_summary",
    "title_url_yake",
    "title_url_rag_sparse",
    "title_url_rag_dense",
    "title_url_rag_sim_doc",
]

def load_txt(file_path):
    res = []
    with open(file_path, 'r', encoding='utf-8') as dfile:
        for line in dfile.readlines():
            if line[-1]=='\n':
                line = line[:-1]
            res.append(line)
    return res

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

def is_empty(text):
    text = text.strip()
    return len(text)==0 or text==""

def process_text(tokenizer, input, context_type):
    form_in, only_pref = prepare_input(tokenizer, input, context_type, inference=True)
    prefix_str = tokenizer.apply_chat_template(form_in, tokenize=False, add_generation_prompt=True)
    return prefix_str, only_pref

def store_jsonl(res, file_path):
    with open(file_path, 'w', encoding='utf-8') as dfile:
        for line in res:
            json.dump(line, dfile, ensure_ascii=False)
            dfile.write('\n')
    print("written %d lines to json file : %s" % (len(res), file_path))

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

def add_ctx_col(input_dict, model_output):
    input_dict["completions"] = model_output

def convert_json_to_tsv(input_json, header):
    final_str=""
    for h in header:
        final_str+=f"{input_json[h]}\t"
    final_str+=f"{input_json['completions']}"
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

def get_partial_prefix(prefix):
    idx = prefix.rfind(" ")
    if idx==-1:
        return "", prefix 
    return prefix[:idx].lstrip(), prefix[idx+1:].lstrip()

def postprocess_response(prefix, responses):
    prefix = prefix.lstrip()
    final_res = []
    for res in responses:
        res = res.replace("##Completion##:", "").strip()
        if prefix[-1]==" ":
            temp_str = prefix + res
            final_res.append(temp_str.strip())
        else:
            partial_prefix, last_word = get_partial_prefix(prefix)
            if res.lower().startswith(last_word.lower()):
                temp_str = partial_prefix + " " + res
                final_res.append(temp_str.strip())
    return final_res

def inference(args):
    print("[worker-%d] using device : %s" % (args.worker_id, args.device))
    # loading tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    tokenizer.pad_token = "<|reserved_special_token_0|>"  # use unk rather than eos token to prevent endless generation
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = "left"
    
    if args.use_trie:
        if args.trie_path is None:
            raise ValueError("Please provide a path to the trie file using --trie_path")
        print("Loading trie from:", args.trie_path)
        # load the token sequences from the trie file
        with open(args.trie_path, 'rb') as f:
            token_sequences = pickle.load(f)
            
        # check if token_sequences is a list of lists or dictionary of lists
        if isinstance(token_sequences, dict):
            token_trie = {k: TokenPrefixSuffixTrie(v) for k, v in token_sequences.items()}
        elif isinstance(token_sequences, list):
            token_trie = TokenPrefixSuffixTrie(token_sequences)

    if args.load_from_hf_checkpoint:
        # loading HF model
        model = AutoModelForCausalLM.from_pretrained(args.model_path, 
                                                torch_dtype=torch.float16,
                                                trust_remote_code=True,
                                                ).to(args.device)
    else:
        # loading PEFT model
        base_model_reload = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct",
                                                trust_remote_code=True,
                                                torch_dtype=torch.float16,
                                                ).to(args.device)
        model = PeftModel.from_pretrained(base_model_reload, args.model_path)
    model = torch.compile(model, mode="reduce-overhead")
    model.config.eos_token_id = tokenizer.eos_token_id
    print("[worker-%d] model loaded successfully" % (args.worker_id))
    model = model.to(args.device).eval()
    print("[worker-%d] total number of parameters : %d" % (args.worker_id, count_parameters(model)))
    output_file = open(args.output_file, 'w', encoding='utf-8')
    prefix_count = args.prefix_count
    print("[worker-%d] prefixes for inference : %d" % (args.worker_id, prefix_count))
    print("[worker-%d] prefixes start index : %d, end_index : %d" % (args.worker_id, args.start_index, args.max_limit))
    args.inference_count = 0
    args.trimmed_prefixes = 0    
    with torch.no_grad():
        print("[worker-%d] starting the inference ..." % args.worker_id)
        batches = []
        records = []
        res = []
        prefix_list = []
        stream = load_text_stream_with_index(args.input_file, args.max_limit, args.start_index)
        with tqdm(total=math.ceil(prefix_count/args.batch_size), desc="inference", unit=" lines", position=0, leave=True) as pbar:
            for status, record, _ in stream: 
                if status==False:
                    break
                curr_row = record.split('\t')
                curr_dict = convert_tsv_to_json(curr_row, args.header)
                records.append(curr_dict)
                curr_ex_prompt, inp_prefix = process_text(tokenizer, curr_dict, args.context_type)
                prefix_list.append(inp_prefix)
                batches.append(curr_ex_prompt)
                # main logic
                if len(batches)==args.batch_size:
                    inputs = tokenizer.batch_encode_plus(batches, padding=True, return_tensors="pt")
                    input_length = 1 if model.config.is_encoder_decoder else inputs["input_ids"].shape[1]
                    
                    logits_processor = None
                    if args.use_trie:
                        if isinstance(token_trie, dict):
                            the_trie = [token_trie[d["docid"]] for d in records]
                        else:
                            the_trie = [token_trie]
                        logit_processor = TrieConstrainedLogitsProcessor(
                            the_trie,
                            prefix_list,
                            tokenizer,
                            input_length,
                            len(batches),
                            alpha=args.alpha,
                            beta=args.beta,
                            initial_bias=args.bias_strength
                        )
                        logits_processor = [logit_processor]
                    
                    generated_ids = model.generate(
                        input_ids=inputs["input_ids"].to(args.device),
                        attention_mask=inputs["attention_mask"].to(args.device),
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        num_beams=args.beam_size,
                        use_cache=True,
                        length_penalty=args.length_penalty,
                        num_return_sequences=args.num_seq,
                        max_new_tokens=args.max_gen_token,
                        logits_processor=logits_processor,
                        early_stopping=True)
                    generated_tokens = generated_ids[:, input_length:]
                    output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    output = list(map(lambda x: x.strip(), output))
                    res = []
                    for idx in range(len(batches)):
                        start_index = idx*args.num_seq
                        end_index = start_index + args.num_seq
                        res.append(output[start_index:end_index])
                    args.inference_count+=len(batches)
                    assert len(batches)==len(res), f"unequal prefix and inference {len(batches)} != {len(res)}"
                    for x, y in zip(records, res):
                        final_y = postprocess_response(x["prefix"], y)
                        output_file.write(json.dumps({"doc_id": x["docid"], "query": x["query"], "prefix": x["prefix"], "completions": [[sz, "SLM:-1"] for sz in final_y]}) +'\n')
                    batches = []
                    res = []
                    records = []
                    torch.cuda.empty_cache()
                    pbar.update(1)

            # process the remaining batches
            if len(batches)>0:
                inputs = tokenizer.batch_encode_plus(batches, padding=True, return_tensors="pt")
                input_length = 1 if model.config.is_encoder_decoder else inputs["input_ids"].shape[1]
                
                logits_processor = None
                if args.use_trie:
                    if isinstance(token_trie, dict):
                        the_trie = [token_trie[d["docid"]] for d in records]
                    else:
                        the_trie = [token_trie]
                    logit_processor = TrieConstrainedLogitsProcessor(
                        the_trie,
                        prefix_list,
                        tokenizer,
                        input_length,
                        len(batches),
                        alpha=args.alpha,
                        beta=args.beta,
                        initial_bias=args.bias_strength
                    )
                    logits_processor = [logit_processor]
                    
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"].to(args.device),
                    attention_mask=inputs["attention_mask"].to(args.device),
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    num_beams=args.beam_size,
                    use_cache=True,
                    length_penalty=args.length_penalty,
                    num_return_sequences=args.num_seq,
                    max_new_tokens=args.max_gen_token,
                    logits_processor=logits_processor,
                    early_stopping=True)
                generated_tokens = generated_ids[:, input_length:]
                output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                output = list(map(lambda x: x.strip(), output))
                res = []
                for idx in range(len(batches)):
                    start_index = idx*args.num_seq
                    end_index = start_index + args.num_seq
                    res.append(output[start_index:end_index])
                args.inference_count+=len(batches)
                assert len(batches)==len(res), f"unequal prefix and inference {len(batches)} != {len(res)}"
                for x, y in zip(records, res):
                    final_y = postprocess_response(x["prefix"], y)
                    output_file.write(json.dumps({"doc_id": x["docid"], "query": x["query"], "prefix": x["prefix"], "completions": [[sz, "SLM:-1"] for sz in final_y]}) +'\n')
                batches = []
                res = []
                records = []
                torch.cuda.empty_cache()
                pbar.update(1)
        
        output_file.close()
        print("[worker-%d] inference done : %d " % (args.worker_id, args.inference_count))

# function to delete the file if it exists
def delete_file_if_exists(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

# function to get the filename and parent directory given the complete file path
def create_worker_file_name(idx, file_path):
    file_name = os.path.basename(file_path)
    parent_dir = os.path.dirname(file_path)
    os.makedirs(parent_dir, exist_ok=True)
    return os.path.join(parent_dir, f"worker-{idx}-{file_name}")

# function to print time delta in human readable format hh:mm:ss
def get_time_delta(time_delta):
    hours, rem = divmod(time_delta.seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Global model configuration
    parser.add_argument('--model_path', type=str, required=True,
                        help='')
    parser.add_argument('--input_file', type=str, required=True,
                        help='')
    parser.add_argument('--output_file', type=str, required=True,
                        help='')
    parser.add_argument('--beam_size', type=int, default=5,
                        help='')
    parser.add_argument('--num_seq', type=int, default=1,
                        help='')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='')
    parser.add_argument('--max_limit', type=int, default=-1,
                        help='')
    parser.add_argument('--max_gen_token', type=int, default=40,
                        help='')
    parser.add_argument('--max_src_token', type=int, default=512,
                        help='')
    parser.add_argument('--length_penalty', type=float, default=1.0,
                        help='')
    parser.add_argument('--load_from_hf_checkpoint', action='store_true', default=False,
                        help='inference from hf checkpoint')
    # if multiple gpus devices are available
    parser.add_argument('--num_workers', type=int, default=1, 
                        help='total number of workers')
    parser.add_argument("--context_type", type=str, choices=exp_variations, default="prefix_only")
    
    parser.add_argument("--use_trie", action="store_true", help="Use trie for constrained decoding")
    parser.add_argument("--trie_path", type=str, default=None, help="Path to the sequence to be added to the trie")
    parser.add_argument("--alpha", type=float, default=0.3, help="Alpha value for the TrieConstrainedLogitsProcessor")
    parser.add_argument("--beta", type=float, default=0.1, help="Beta value for the TrieConstrainedLogitsProcessor")
    parser.add_argument("--bias_strength", type=float, default=30, help="Bias strength for the TrieConstrainedLogitsProcessor")                                    
    args = parser.parse_args()

    global_start_time = datetime.utcnow()
    print("[global] argument passed :")
    for arg in vars(args):
        print("[global] %s - %s" % (arg, getattr(args, arg)))
    print('--'*30)
    args.header = extract_header(args.input_file)
    args.global_prefix_count = get_prefix_count(args.header, args.input_file, args.max_limit, 0)
    print("[global] total prefix count : %d" % args.global_prefix_count)
    inference_count = 0
    trimmed_prefix_count = 0
    if args.num_workers==1:
        args.worker_id = 0
        # skip the header
        args.start_index = 1
        args.prefix_count = args.global_prefix_count
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        inference(args)
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
            worker_config.start_index = idx*bucket_size if idx>0 else 1
            worker_config.max_limit = min(worker_config.start_index + bucket_size, args.global_prefix_count)
            worker_config.prefix_count = (worker_config.max_limit - worker_config.start_index)
            worker_config.device = f"cuda:{idx}" if torch.cuda.is_available() else "cpu"
            worker_config.output_file = create_worker_file_name(idx, os.path.abspath(args.output_file))
            worker_configs.append(worker_config)
        # spawning the process
        worker_processes = []
        print("[global] spawning %d inference processes" % args.num_workers)
        for idx in range(args.num_workers):
            p = mp.Process(target=inference, args=(worker_configs[idx],))
            p.start()
            worker_processes.append(p)
        # wait for all the process to finish
        for p in worker_processes:
            p.join()
        
        # merge intermediate outputs to final file 
        final_output_file = open(args.output_file, 'w', encoding='utf-8')
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
    print("[global] completed %d inference in %s" % (args.global_prefix_count, get_time_delta(time_delta)))