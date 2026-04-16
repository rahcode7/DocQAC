from utils_global_tries import GlobalTries
import pickle
import sys 
# Load the trie from a file
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


# Testing code below 
# python query-auto-suggest-share\src\code\tries\parallel_mpc_inference_nlg.py
# Sample output - [['blurry eye', 'MT:1000.0'], ['bluetooth special interest group', 'MT:1000.0'], ['blues brothers', 'MT:1000.0'], ['blues', 'MT:1000.0'], ['bluedevil', 'MT:1000.0'], ['bluecross blue shield', 'MT:1000.0'], ['bluecross', 'MT:1000.0'], ['bluebird', 'MT:1000.0'], ['blue tooth', 'MT:1000.0'], ['blue shield blue', 'MT:1000.0']]

# Tries download - 

if __name__=="__main__":
    # Load tries
    main_trie_path = "datasets/outputs/global-tries/main.mpc"
    suffix_trie_path = "datasets/outputs/global-tries/suffix.mpc"
    query_completion,suffix_completion = load_tries(main_trie_path,suffix_trie_path)

    # Get global completions
    prefix = "bl"
    k_completions = 10
    trie_completions = global_inference(query_completion,suffix_completion,prefix,k_completions,suffix_context=2)
    print(trie_completions)

    print("Continue execution")
    #sys.exit(0)