import math
import yake
import torch
import struct
import pickle
import argparse
import marisa_trie
import numpy as np
import pandas as pd
from tqdm.auto import tqdm as tq
from torch.utils.data import Dataset
from transformers import LogitsProcessor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ..tries.utils_global_tries import GlobalTries
from rag_utils import get_chunks_sparse, rag_sparse_loader, get_chunks_dense, rag_loader_dense, similar_doc_loader,get_similar_docs,get_chunks_similar_dense
tq.pandas()


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
    
    def get_chidren(self, prefix_tokens):
        """
        prefix: prefix token ids  [id1,id2]
        returns: set of token ids that are possible next [id3,id4]
        """
        
        prefix_len = len(prefix_tokens)
        packed_prefix_seq = b''.join(struct.pack(self.fmt, int(token_id)) for token_id in prefix_tokens)
        prefix_hex = packed_prefix_seq.hex()
        completions = self.trie.keys(prefix_hex) 

        completions2 = set()
        for key_hex in completions:
            key_bytes = bytes.fromhex(key_hex)
            
            jk = prefix_len * struct.calcsize(self.fmt)
            next_node = struct.unpack(self.fmt, key_bytes[jk:jk+struct.calcsize(self.fmt)])[0]
            completions2.add(next_node)
            
        return sorted(list(completions2))
    
    
class TrieConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(self, trie, prefix,tokenizer, input_ids, initial_bias=30.0, alpha=0.0,beta=0.0,max_bias=100.0):
        self.trie_list = trie
        self.prefix = prefix  # growing prefix
        self.tokenizer = tokenizer
        
        self.initial_bias = initial_bias
        self.alpha = alpha
        self.max_bias = max_bias
        self.beta = beta 
        self.input_ids = input_ids
        self.query_id = tokenizer.convert_tokens_to_ids("query:")
        self.batch_size = input_ids.shape[0]  # Number of sequences in the batch
        self.padding_id = tokenizer.pad_token_id
        
        
    def remove_trailing_pad(self, input_lst):
        """
        Removes trailing padding tokens from the input list.
        """
        if isinstance(input_lst, list):
            while input_lst and input_lst[-1] == self.padding_id:
                input_lst.pop()
        return input_lst

    def __call__(self, decoder_input_ids, scores):
        beams = decoder_input_ids.shape[0]// self.batch_size  # Number of beams per batch
        for i in range(len(decoder_input_ids)):
            if len(self.trie_list) > 1:
                trie = self.trie_list[i//beams]
            else:
                trie = self.trie_list[0]
            current_prefix = self.input_ids[i//beams].tolist()
            
            # find query token id in the current prefix
            try:
                current_prefix = current_prefix[current_prefix.index(self.query_id) + 1:]
            except ValueError:
                current_prefix = current_prefix[0:] 
            current_prefix = self.remove_trailing_pad(current_prefix)
            current_prefix = current_prefix[:-1] + [40000] + decoder_input_ids[i, 1:].tolist()
            
            allowed_tokens = trie.get_chidren(current_prefix)
            # allowed_tokens.append([tokenizer.eos_token_id])
            # print(f"current prefix and allowed",allowed_tokens)

            if not allowed_tokens:
                # return scores  # no bias if no trie guidance)
                continue
            
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
            # for token_id in allowed
            # 
            # _tokens:
            #     scores[i, token_id] += annealed_bias

            # Increase the bias with depth
            #annealed_bias = min(self.max_bias, self.initial_bias * math.exp(self.alpha * depth))

            
        return scores


def suffix_encoder(tokenizer, text, max_length, batching = False, prev_space = True):
    if(batching):
        for i in range(len(text)):
            if(not prev_space[i]):
                text[i] = "«" + text[i]
    else:
        if(not prev_space):
            text = "«" + text
    encoded = tokenizer(text, padding="max_length", max_length=64, truncation=True, return_tensors='pt')
    if(batching):
        for i in range(len(encoded['input_ids'])):
            if(not prev_space[i]):
                encoded['input_ids'][i] = torch.cat((encoded['input_ids'][i][1:], torch.tensor([tokenizer.pad_token_id])))
                encoded['attention_mask'][i] = torch.cat((encoded['attention_mask'][i][1:], torch.tensor([0])))
        return encoded
    if(not prev_space):
        encoded['input_ids'][0] = torch.cat((encoded['input_ids'][0][1:], torch.tensor([tokenizer.pad_token_id])))
        encoded['attention_mask'][0] = torch.cat((encoded['attention_mask'][0][1:], torch.tensor([0])))
    return encoded

def suffix_decoder(tokenizer, encoded):
    text = tokenizer.decode(torch.cat((torch.tensor([673]), encoded), dim=0), skip_special_tokens=True)
    text = text[1:]
    # if(len(text)==0):
    #     print("Empty text")
    #     print(encoded)
    return text


def prefix_encoder(tokenizer, text, max_length, batch = False):
    if(batch):
        for i in range(len(text)):
            if(text[i][-1]==" "):
                text[i] = text[i][:-1] + "<tspace>"
    else:
        if(text[-1]==" "):
            text = text[:-1] + "<tspace>"
    encoded = tokenizer(text, padding="max_length", max_length=max_length, truncation=True, return_tensors='pt')
    return encoded

def merge_prefix_suffix(prefix, suffix):
    if(len(suffix)>0 and len(prefix)>0 and suffix[0] == " " and prefix[-1] == " "):
        return prefix[:-1] + suffix
    else:
        return prefix + suffix
    
def load_trie_from_file(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def global_inference(query_completion,suffix_completion,prefix,k_completions=10,suffix_context=2):
    prefix = prefix.lstrip().lower()
    completions = GlobalTries.get_completions(query_completion, suffix_completion, prefix, 
                                    k_completions, suffix_context)
    return completions

def load_tries(main_trie_path,suffix_trie_path):
    query_completion = load_trie_from_file(main_trie_path)
    suffix_completion = None
    if suffix_trie_path is not None:
        suffix_completion = load_trie_from_file(suffix_trie_path)
    
    return query_completion,suffix_completion  


class AutocompleteDataset(Dataset):
    def __init__(self,
        data_path,
        doc_path,
        tokenizer, 
        tkmax_length=512,
        infer=False,
        in_type="no_doc",
        suffix_trie_path="datasets/outputs/global-tries/suffix.mpc",
        main_trie_path="datasets/outputs/global-tries/main.mpc",
        k_comp = 10
        ):

        self.tokenizer = tokenizer
        self.in_type = in_type

        self.max_length = tkmax_length
        self.infer = infer

        language = "en"
        max_ngram_size = 3
        deduplication_thresold = 0.9
        deduplication_algo = 'seqm'
        windowSize = 1
        numOfKeywords = 50

        self.extractor = yake.KeywordExtractor(
            lan=language, 
            n=max_ngram_size, 
            dedupLim=deduplication_thresold, 
            dedupFunc=deduplication_algo, 
            windowsSize=windowSize, 
            top=numOfKeywords, 
            features=None
            )
        # preprocessing

        if in_type != "no_doc":
            
            ctx_len = 352
            
            if "trie" in in_type:
                self.query_completion,self.suffix_completion = load_tries(main_trie_path,suffix_trie_path)
                self.k_comp = k_comp
                
                ctx_len = 300

            if ".tsv" in data_path:
                self.data = pd.read_csv(data_path, sep="\t")
            else:
                self.data = pd.read_csv(data_path)

            self.doc = pd.read_csv(doc_path)
            self.doc['heading'] = self.doc['heading'].fillna('')

            if self.in_type == "yake":
                if ".tsv" in doc_path:
                    save_path = doc_path.split(".tsv")[0] + "_yake.tsv"
                else:
                    save_path = doc_path.split(".csv")[0] + "_yake.csv"
                
                try:
                    if ".tsv" in save_path:
                        self.doc = pd.read_csv(save_path, sep="\t")
                    else:
                        self.doc = pd.read_csv(save_path)
                        
                    self.doc["yake"] = self.doc["yake"].fillna("")
                    self.doc["yake"] = self.doc["yake"].apply(lambda x: self.trunc_text(x, ctx_len)["text"])
                    self.doc["url"] = self.doc["doc_url"].apply(lambda x: self.trunc_text(x, 32)["text"])
                    self.doc["heading"] = self.doc["heading"].apply(lambda x: " ".join(x.split(" ")[:32]))
                except:
                    self.doc["yake"] = self.doc["body"].progress_apply(lambda x: self.yake_extract(x)["yake"])
                    self.doc["yake"] = self.doc["yake"].apply(lambda x: " <eok> ".join(x))
                    self.doc["yake"] = self.doc["yake"].apply(lambda x: self.trunc_text(x, ctx_len)["text"])
                    self.doc["url"] = self.doc["doc_url"].apply(lambda x: self.trunc_text(x, 32)["text"])
                    self.doc["heading"] = self.doc["heading"].apply(lambda x: " ".join(x.split(" ")[:32]))
                    
                    if ".tsv" in save_path:
                        self.doc.to_csv(save_path, sep="\t", index=False)
                    else:
                        self.doc.to_csv(save_path, index=False)


            if self.in_type == "full_doc":
                ctx_len = 384
                if "trie" in in_type:
                    ctx_len = 330
                self.doc["url"] = self.doc["doc_url"].apply(lambda x: self.trunc_text(x, 32)["text"])
                self.doc["body"] = self.doc["body"].apply(lambda x: self.trunc_text(x, ctx_len)["text"])
                self.doc["heading"] = self.doc["heading"].apply(lambda x: " ".join(x.split(" ")[:32]))
                # self.
            
            if self.in_type == "summary":
                #print("*"*100)
                self.doc["url"] = self.doc["doc_url"].apply(lambda x: self.trunc_text(x, 32)["text"])
                self.doc["body"] = self.doc["body"].apply(lambda x: self.trunc_text(x, ctx_len)["text"])
                self.doc["heading"] = self.doc["heading"].apply(lambda x: " ".join(x.split(" ")[:32]))
            
            if self.in_type == "url":
                ctx_len = 384
                if "trie" in in_type:
                    ctx_len = 330
                    
                self.doc["body"] = self.doc["doc_url"].apply(lambda x: self.trunc_text(x, ctx_len)["text"])
                self.doc["heading"] = self.doc["heading"].apply(lambda x: " ".join(x.split(" ")[:32]))
            
            if self.in_type == "url_doc":
                self.doc["url"] = self.doc["doc_url"].apply(lambda x: self.trunc_text(x, 32)["text"])
                self.doc["body"] = self.doc["body"].apply(lambda x: self.trunc_text(x, ctx_len)["text"])
                self.doc["heading"] = self.doc["heading"].apply(lambda x: " ".join(x.split(" ")[:32]))

            if "rag" in self.in_type:

                if "dense" in self.in_type:
                    self.embedding, self.text_splitter = rag_loader_dense(200, 30)
                    if "sim" in self.in_type:
                        pickle_path="datasets/rag/similar_docs/similar_docs.pkl"
                        self.sim_doc_dict = similar_doc_loader(pickle_path)
                else:

                    self.text_splitter = rag_sparse_loader()
                
                self.doc["url"] = self.doc["doc_url"].progress_apply(lambda x: self.trunc_text(x, 32)["text"])
                # self.doc["url"] = self.doc["doc_url"].progress_apply(lambda x: " ".join(x.split(" ")[:32]))
                self.doc["heading"] = self.doc["heading"].progress_apply(lambda x: " ".join(x.split(" ")[:32]))

            #self.df = self.data.merge(self.doc, on='docid')
            # 
            self.df = pd.merge(self.data,self.doc,on='docid')

        else:
            if ".tsv" in data_path:
                self.data = pd.read_csv(data_path, sep="\t")
            else:
                self.data = pd.read_csv(data_path)
            self.df = self.data
        
        self.ctx_len = ctx_len
        
        # first 1000 examples
        # self.df = self.df[:50]

    def yake_extract(self, text):
        keywords = self.extractor.extract_keywords(str(text))
        keywords = [str(k[0]) for k in keywords]
        return {"yake": keywords}
    
    def trunc_text(self, text, max_length):
        text = text.split(" ")
        text = text[:max_length]
        tokenized = self.tokenizer(" ".join(text), max_length=max_length, truncation=True, return_tensors='pt', padding=False)
        text = self.tokenizer.decode(tokenized['input_ids'][0], skip_special_tokens=True)
        return {"text": text}

    def __len__(self):
        # return 100
        return len(self.df)

    def __getitem__(self, idx):
        curr_eg = self.df.iloc[idx]
        # if self.infer == True :
        #     # i think both can be of same format during inference
        #     input_text = curr_sentence.split("\t")[0]
        #     target_text = curr_sentence.split("\t")[1]
        #     return input_text, target_text

        if(self.infer):

            trie_text = ""
            addon = 0

            if "trie" in self.in_type:
                trie_completions = global_inference(self.query_completion,self.suffix_completion,curr_eg["prefix"],self.k_comp,suffix_context=2)
                trie_completions = [x[0] for x in trie_completions]
                trie_text = ", ".join(trie_completions)
                trie_text = " trie: " + trie_text
                addon = 50

            if self.in_type in ["full_doc", "summary"]:
                ctx = curr_eg["body"]
                heading = curr_eg["heading"]
                url = curr_eg["url"]
                input = "title: " + str(heading) + " url: " + str(url) + " context: " + str(ctx) + trie_text + " query: " + str(curr_eg["prefix"])
                self.max_length = 512
            
            elif self.in_type == "url":
                ctx = curr_eg["body"]
                heading = curr_eg["heading"]
                input = "title: " + str(heading) + " url: " + str(ctx)+ trie_text + " query: " + str(curr_eg["prefix"])
                self.max_length = 192 + addon

            elif self.in_type == "no_doc":
                # print(curr_eg)
                if trie_text != " ":
                    trie_text += " query: "
                    
                input = trie_text + str(curr_eg["prefix"])

            elif self.in_type == "url_doc":
                ctx = curr_eg["body"]
                heading = curr_eg["heading"]
                url = curr_eg["url"]
                input = "title: " + str(heading) + " url: " + str(url) + " context: " + str(ctx) + trie_text + " query: " + str(curr_eg["prefix"])
                self.max_length = 512

            elif self.in_type == "yake":
                ctx = curr_eg["yake"]
                url = curr_eg["url"]
                heading = curr_eg["heading"]
                input = "title: " + str(heading) + " url: " + str(url) + " context: " + str(ctx) + trie_text + " query: " + str(curr_eg["prefix"])
                self.max_length = 512

            elif "rag" in self.in_type:
                if "dense" in self.in_type:
                    if "sim" in self.in_type:
                        sim_doc_list = get_similar_docs(curr_eg["docid"], self.sim_doc_dict)
                        chunks = get_chunks_similar_dense(curr_eg["prefix"] ,curr_eg["docid"], curr_eg["body"], sim_doc_list, self.embedding, self.text_splitter, k=40)
                    else:
                        chunks = get_chunks_dense(curr_eg["prefix"], curr_eg["docid"], curr_eg["body"], self.embedding, self.text_splitter)
                else:
                    chunks = get_chunks_sparse(curr_eg["prefix"], curr_eg["body"], self.text_splitter)
                chunks_len = len(chunks)
                budget = (self.ctx_len - chunks_len)//chunks_len
                final_cated = ""
                for c in chunks:
                    # print(c)

                    if c:
                        new_c = " <eok> ".join(c)
                        final_cated += self.trunc_text(new_c, budget)["text"] + " <eok> "
                        # final_cated += " ".join(new_c[:budget]) + " <eok> " # to make it faster
                # remove last <eok>
                final_cated = final_cated[:-7]
                input = "title: " + str(curr_eg["heading"]) + " url: " + str(curr_eg["url"]) + " context: " + str(final_cated) + trie_text + " query: " + str(curr_eg["prefix"])
                self.max_length = 512

            input_encoded = prefix_encoder(self.tokenizer, input, max_length=self.max_length)


            return input_encoded, curr_eg["prefix"], curr_eg["query"], curr_eg["docid"]

        r = np.random.randint(1, curr_eg["query_length"])
        input_text = curr_eg["query"][:r]
        target_text = curr_eg["query"][r:]
        
        trie_text = ""
        addon = 0

        if "trie" in self.in_type:
            trie_completions = global_inference(self.query_completion,self.suffix_completion,input_text,self.k_comp,suffix_context=2)
            trie_completions = [x[0] for x in trie_completions]
            trie_text = ", ".join(trie_completions)
            trie_text = "trie: " + trie_text
            addon = 50
            
        if self.in_type in ["full_doc", "summary"]:
            # print(curr_eg.keys())
            input_text = "title: " + str(curr_eg["heading"]) + " url: " + str(curr_eg["url"]) + " context: " + str(curr_eg["body"]) + trie_text + " query: " + str(input_text)
            self.max_length = 512
        
        elif self.in_type == "url":
            input_text = "title: " + str(curr_eg["heading"]) + " url: " + str(curr_eg["body"]) + trie_text + " query: " + str(input_text)
            self.max_length = 192 + addon
        
        elif self.in_type == "url_doc":
            input_text = "title: " + str(curr_eg["heading"]) + " url: " + str(curr_eg["url"]) + " context: " + str(curr_eg["body"]) + trie_text + " query: " + str(input_text)
            self.max_length = 512

        elif self.in_type == "yake":
            input_text = "title: " + str(curr_eg["heading"]) + " url: " + str(curr_eg["url"]) + " context: " + str(curr_eg["yake"]) + trie_text + " query: " + str(input_text)
            self.max_length = 512
        
        elif self.in_type == "no_doc":
            if trie_text != " ":
                    trie_text += " query: "
                    
            input_text = trie_text + str(input_text)

        elif "rag" in self.in_type:
            if "dense" in self.in_type:
                if "sim" in self.in_type:
                    sim_doc_list = get_similar_docs(curr_eg["docid"], self.sim_doc_dict)
                    chunks = get_chunks_similar_dense(input_text ,curr_eg["docid"], curr_eg["body"], sim_doc_list, self.embedding, self.text_splitter, k=40)
                else:
                    chunks = get_chunks_dense(input_text, curr_eg["docid"], curr_eg["body"], self.embedding, self.text_splitter)
            else:
                chunks = get_chunks_sparse(input_text, curr_eg["body"], self.text_splitter)
                
            chunks_len = len(chunks)
            budget = (self.ctx_len - chunks_len)//chunks_len
            final_cated = ""
            for c in chunks:
                # print(c)

                if c:
                    new_c = " <eok> ".join(c)
                    final_cated += self.trunc_text(new_c, budget)["text"] + " <eok> "
                    # final_cated += " ".join(new_c[:budget]) + " <eok> " # to make it faster
            # remove last <eok>
            final_cated = final_cated[:-7]
            input_text = "title: " + str(curr_eg["heading"]) + " url: " + str(curr_eg["url"]) + " context: " + str(final_cated) + trie_text + " query: " + str(input_text)
            self.max_length = 512

        

        # elif self.in_type == "no_doc":
        #     input_text = "query: " + input_text

        # print("input = ", [input_text])
        # print("target = ", [target_text])


        input_encoded = prefix_encoder(self.tokenizer, input_text, max_length=self.max_length)
        target_encoded = suffix_encoder(self.tokenizer, target_text, max_length=self.max_length, prev_space = (input_text[-1]==" "))

        
        # print("input = ", [input_text])
        # print("target = ", [target_text])
        # print("target decoded = ", [suffix_decoder(self.tokenizer, target_encoded['input_ids'][0])])
        return input_encoded, target_encoded
    
    def preprocess_text(self, text):
        text = text.strip().lower()
        text = text.replace("<eou>", "<|EOU|>")
        return text