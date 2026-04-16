# Get list of unique words from document
import datrie
import string
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
import re
import collections
import string

def unique_words(doc:str):

    #  lowercase
    doc =  doc.lower()

    doc_arr = doc.lower().split(" ")
    print(len(doc_arr))
    print(len(list(set(doc_arr))))

    # How to handle special character for 'Cruiser#3' and 'Tahoe#2' and 'CX-3#2' ????  

    return list(set(doc_arr))

def trie_doc_query(queries : list[str], poploc : list[int]):
    """Returns a trie object for all the queries"""

    trie = datrie.Trie(string.printable)

    # for query in queries:
    #     #print(query)
    #     # Query level
    #     #uquery = unicode(query, "utf-8")
    #     trie[query] = [0,0] # Local and global popularity

    for query,popl in zip(queries,poploc):
        #print(query,popl)
        trie[query] = [popl,0] # Local and global popularity

    return trie

def tokenize(string):
    return re.findall(r'\w+', string.lower())

def generate_n_grams(sentence, n):
    unigrams = ngrams(sentence.split(), n)
    return [' '.join(unigram) for unigram in unigrams] # Convert tuple to string and return

def count_n_grams(ngrams):
    ngram_counts = collections.Counter(ngrams)
    return ngram_counts
