from queryblazer import *
import argparse
import json
import pickle

if __name__ == "__main__":

    config = Config(branch_factor=30, beam_size=100, topk=100, length_limit=100, precompute=True)
    qbz = QueryBlazer(encoder="encoder.fst", model="ngram.fst", config=config)
    qbz.SavePrecomputed('precomputed.bin')
