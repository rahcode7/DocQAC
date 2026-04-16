import argparse
from copy import deepcopy
import os
import sys
import json
#from tqdm import tqdm
import time 

import pickle
from utils import QueryCompletion, preprocess, load_text_stream, is_empty, create_suffixes, check_and_create_path, get_line_count
import re
import math
from datetime import datetime
import multiprocessing as mp
import csv
from collections import OrderedDict
sys.setrecursionlimit(1000)

class GlobalTries:

    @staticmethod
    def get_completions(main_trie, suffix_trie, prefix, k_completions=100, suffix_context=2):
        res = []
        #main_completions = main_trie.find_completions(prefix,k_completions=20)
        main_completions = main_trie.find_completions(prefix,k_completions=k_completions)

        #print(main_completions)
        main_completions_dict = {}
        #print(main_completions)
        for z in main_completions:
            if z[0] in main_completions_dict:
                continue
            main_completions_dict[z[0]]=1
            # adding prefix "MT" to denote main trie completions
            res.append([z[0], "MT:%s" % str(z[1])])

        if suffix_trie is not None and len(res)<k_completions:
            backfill = k_completions - len(res)
            prefix_tokens = prefix.strip().split(' ')
            ends_with_space = " " if prefix[-1]==" " else ""
            suffix_completions = []
            suffix_completions_dict = {}
            
            # minimum words to consider during suffix match
            for idx in range(suffix_context, len(prefix_tokens)):
                suffix = " ".join(prefix_tokens[-idx:]) + ends_with_space
                partial_prefix = " ".join(prefix_tokens[:len(prefix_tokens)-idx])
                for temp_completions in suffix_trie.find_completions(suffix):
                    full_completion = partial_prefix + " " + temp_completions[0]
                    if full_completion in main_completions_dict or full_completion in suffix_completions_dict:
                        continue
                    suffix_completions_dict[full_completion] = 1
                    suffix_completions.append((full_completion, temp_completions[1]))
                # print("suffix completions : %d" % len(suffix_completions))
            suffix_completions = sorted(suffix_completions, key=lambda x: x[1], reverse=True)[:backfill]
            if len(suffix_completions)>0:
                for z in suffix_completions:
                    # adding prefix "ST" to denote suffix trie completions
                    res.append([z[0], "ST:%s" % str(z[1])])

        # Remove duplicates
        unique_k = list(OrderedDict.fromkeys(map(tuple, res)))
        # Convert back to list of lists
        unique_k = list(map(list, unique_k))
        
        return res[0:k_completions]
