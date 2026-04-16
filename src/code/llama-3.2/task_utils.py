import re
import math
import torch
import struct
import marisa_trie
from transformers import LogitsProcessor

class TokenPrefixSuffixTrie:
    def __init__(self, token_sequences):
        self.fmt = "<I"
        byte_sequences = []
        for seq in token_sequences:
            packed_seq = b''.join(struct.pack(self.fmt, int(token_id)) for token_id in seq)
            byte_sequences.append(packed_seq)
        
        hex_keys = [b.hex() for b in byte_sequences]
        self.trie = marisa_trie.BytesTrie(zip(hex_keys, byte_sequences))

    def get_completions(self, prefix_tokens):
        """
        prefix: prefix token ids  [id1,id2]
        returns: set of token ids completions [[id1,id2, id3,id4], [id1,id2, id5,id6]]
        """

        packed_prefix_seq = b''.join(struct.pack(self.fmt, int(token_id)) for token_id in prefix_tokens)
        prefix_hex = packed_prefix_seq.hex()
        completions = self.trie.keys(prefix_hex) 

        completions2 = []
        # print(completions)
        for key_hex in completions:
            # Convert hex string back to bytes
            key_bytes = bytes.fromhex(key_hex)
            token_ids_back = []
            # To convert back to token IDs:
            for i in range(0, len(key_bytes), struct.calcsize(self.fmt)):
                token_ids_back.append(struct.unpack(self.fmt, key_bytes[i:i+struct.calcsize(self.fmt)])[0])
            completions2.append(token_ids_back)
        return completions2

    def get_children(self, prefix_tokens):
<<<<<<< HEAD:src/code/llama-3.2-constrained/task_utils.py
            """
            prefix: prefix token ids  [id1,id2]
            returns: set of token ids that are possible next [id3,id4]
            """
            
            prefix_len = len(prefix_tokens)
            packed_prefix_seq = b''.join(struct.pack(self.fmt, int(token_id)) for token_id in prefix_tokens)
            prefix_hex = packed_prefix_seq.hex()
            completions = self.trie.keys(prefix_hex) 
=======
        """
        prefix: prefix token ids  [id1,id2]
        returns: set of token ids that are possible next [id3,id4]
        """
        
        prefix_len = len(prefix_tokens)
        packed_prefix_seq = b''.join(struct.pack(self.fmt, int(token_id)) for token_id in prefix_tokens)
        prefix_hex = packed_prefix_seq.hex()
        completions = self.trie.keys(prefix_hex) 
>>>>>>> 7420f23 (addd):src/code/llama-3.2/task_utils.py

            completions2 = set()
            for key_hex in completions:
                key_bytes = bytes.fromhex(key_hex)
                
                jk = prefix_len * struct.calcsize(self.fmt)

                data = key_bytes[jk:jk+struct.calcsize(self.fmt)]
                if len(data) < struct.calcsize(self.fmt):
                    return []  # or handle gracefully
                next_node = struct.unpack(self.fmt, data)[0]
                #next_node = struct.unpack(self.fmt, key_bytes[jk:jk+struct.calcsize(self.fmt)])[0]
                completions2.add(next_node)
                
            return sorted(list(completions2))
    

class TrieConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(self, trie, prefix, tokenizer, seq_len, batch_size, initial_bias=30.0, alpha=0.0,beta=0.0,max_bias=100.0):
        self.trie_list = trie
        self.prefix = prefix  # growing prefix
        self.tokenizer = tokenizer
        
        self.initial_bias = initial_bias
        self.alpha = alpha
        self.max_bias = max_bias
        self.beta = beta
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.padding_id = tokenizer.pad_token_id
        
        self.inid_attm = tokenizer(prefix, add_special_tokens=False)
        self.prefix_ids = self.inid_attm.input_ids

    def remove_trailing_pad(self, input_lst):
        """
        Removes trailing padding tokens from the input list.
        """
        if isinstance(input_lst, list):
            while input_lst and input_lst[-1] == self.padding_id:
                input_lst.pop()
        return input_lst

    def __call__(self, input_ids, scores):
        beams = input_ids.shape[0]// self.batch_size  # Number of beams per batch
        for i in range(len(input_ids)):
            if len(self.trie_list) > 1:
                trie = self.trie_list[i//beams]
            else:
                trie = self.trie_list[0]
            current_prefix = self.prefix_ids[i//beams].copy()  # Get the prefix for the current beam
            
            # find query token id in the current prefix
            current_suffix = input_ids[i].tolist()[self.seq_len:]
            
            current_prefix = current_prefix + [2**20] + current_suffix

            allowed_tokens = trie.get_children(current_prefix)
            # allowed_tokens.append([tokenizer.eos_token_id])
            # print(f"current prefix and allowed",allowed_tokens)

            if not allowed_tokens:
                # return scores  # no bias if no trie guidance)
                return scores
            
             # Decreasing the bias with depth
            depth = len(current_prefix)
            
            # Model decides whether to go with trie 
            beam_depth =  i % beams 
            # print(self.alpha,self.beta)
            annealed_bias = self.initial_bias * math.exp(-self.alpha * depth)  * math.exp(-self.beta * beam_depth)
            vocab_size = scores.shape[-1]
            device = scores.device
            mask = torch.ones(vocab_size, dtype=torch.bool, device=device)
            mask[allowed_tokens] = False
            scores[i, mask] -= annealed_bias

           
            #0.3 # 0.0 - no bias 
            #annealed_bias = self.initial_bias * math.exp(-alpha * depth) 
            # for token_id in allowed_tokens:
            #     scores[i, token_id] += annealed_bias

            # Increase the bias with depth
            #annealed_bias = min(self.max_bias, self.initial_bias * math.exp(self.alpha * depth))

            
        return scores
    

def normalize_text(text):
    text = text.lower().strip()
    text = re.sub('\s+', ' ', text)
    return text

def valid_record(record):
    for key, value in record.items():
        if value is None:
            return False
    return True

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

def prepare_input(tokenizer, input, context_type, inference=False):
    messages = []
    # adding instructions
    messages.append({"role": "system", "content": "You are an intelligent AI system trained to generate the completion given the prefix and context."})
    prefix = tokenizer_budget(tokenizer, input["prefix"], 40)
    if not inference:
        completion = tokenizer_budget(tokenizer, input["suffix"], 40, truncation_side='right')
    # user prompts
    if context_type == "title_doc":
        doc_content = tokenizer_budget(tokenizer, input["content"], 300, truncation_side='right')
        messages.append({"role": "user", "content": f"##Title: {doc_title}, ##Content##: {doc_content}, ##Prefix##: {prefix} ##End##"})
    elif context_type == "title_url":
        doc_title = tokenizer_budget(tokenizer, input["title"], 20, truncation_side='right')
        doc_url = tokenizer_budget(tokenizer, input["url"], 20)
        messages.append({"role": "user", "content": f"##URL: {doc_url}, ##Title: {doc_title}, ##Prefix##: {prefix} ##End##"})
    elif context_type == "title_url_doc":
        doc_title = tokenizer_budget(tokenizer, input["title"], 20, truncation_side='right')
        doc_url = tokenizer_budget(tokenizer, input["url"], 20)
        doc_content = tokenizer_budget(tokenizer, input["content"], 300, truncation_side='right')
        messages.append({"role": "user", "content": f"##URL: {doc_url}, ##Title: {doc_title}, ##Content##: {doc_content}, ##Prefix##: {prefix} ##End##"})
    elif context_type == "title_url_summary":
        doc_title = tokenizer_budget(tokenizer, input["title"], 20, truncation_side='right')
        doc_url = tokenizer_budget(tokenizer, input["url"], 20)
        doc_summary = tokenizer_budget(tokenizer, input["summary"], 300, truncation_side='right')
        messages.append({"role": "user", "content": f"##URL: {doc_url}, ##Title: {doc_title}, ##Content##: {doc_summary}, ##Prefix##: {prefix} ##End##"})
    elif context_type == "title_url_yake":
        doc_title = tokenizer_budget(tokenizer, input["title"], 20, truncation_side='right')
        doc_url = tokenizer_budget(tokenizer, input["url"], 20)
        doc_yake = tokenizer_budget(tokenizer, input["yake_keywords"], 300, truncation_side='right')
        messages.append({"role": "user", "content": f"##URL: {doc_url}, ##Title: {doc_title}, ##Content##: {doc_yake}, ##Prefix##: {prefix} ##End##"})
    elif context_type == "title_url_rag_sparse":
        doc_title = tokenizer_budget(tokenizer, input["title"], 20, truncation_side='right')
        doc_url = tokenizer_budget(tokenizer, input["url"], 20)
        processed_chunks = tokenizer_budget(tokenizer, input["rag_sparse"], 300, truncation_side='right')
        messages.append({"role": "user", "content": f"##URL: {doc_url}, ##Title: {doc_title}, ##Content##: {processed_chunks}, ##Prefix##: {prefix} ##End##"})
    elif context_type == "title_url_rag_dense":
        doc_title = tokenizer_budget(tokenizer, input["title"], 20, truncation_side='right')
        doc_url = tokenizer_budget(tokenizer, input["url"], 20)
        processed_chunks = tokenizer_budget(tokenizer, input["rag_dense"], 300, truncation_side='right')
        messages.append({"role": "user", "content": f"##URL: {doc_url}, ##Title: {doc_title}, ##Content##: {processed_chunks}, ##Prefix##: {prefix} ##End##"})
    elif context_type == "title_url_rag_sim_doc":
        doc_title = tokenizer_budget(tokenizer, input["title"], 20, truncation_side='right')
        doc_url = tokenizer_budget(tokenizer, input["url"], 20)
        processed_chunks = tokenizer_budget(tokenizer, input["rag_sim_doc"], 300, truncation_side='right')
        messages.append({"role": "user", "content": f"##URL: {doc_url}, ##Title: {doc_title}, ##Content##: {processed_chunks}, ##Prefix##: {prefix} ##End##"})
    else:
        messages.append({"role": "user", "content": f"##Prefix##: {prefix} ##End##"})
    
    # disable completion for inference
    if not inference:
        # assistant prompt
        messages.append({"role": "assistant", "content": f"##Completion##: {completion}"})
    return messages, prefix