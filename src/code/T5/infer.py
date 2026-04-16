import pandas as pd
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
import argparse
import os
from accelerate import Accelerator
import tqdm
import numpy as np
import sys
import json
import pickle
sys.path.append('./code')
from utils import AutocompleteDataset, merge_prefix_suffix, prefix_encoder, suffix_decoder, suffix_encoder, TrieConstrainedLogitsProcessor, TokenPrefixSuffixTrie
import time

def collate_fn(batch):
    """
    Handles both inference (inputs, prefix, query, docid) and train (inputs, targets)
    - expects input dicts that have tensors with leading dim 1 (from tokenizer return_tensors='pt')
    """
    # detect format by length of tuple returned by dataset.__getitem__
    if len(batch[0]) == 2:
        # training mode: (input_encoded, target_encoded)
        inputs_list, targets_list = zip(*batch)  # tuples of length B

        batched_inputs = {}
        batched_targets = {}

        # keys assumed same for all items
        for k in inputs_list[0].keys():
            # each inputs_list[i][k] is e.g. tensor shape [1, L]
            tensors = [inp[k].squeeze(0) for inp in inputs_list]
            batched_inputs[k] = torch.stack(tensors, dim=0)

        for k in targets_list[0].keys():
            tensors = [t[k].squeeze(0) for t in targets_list]
            batched_targets[k] = torch.stack(tensors, dim=0)

        return batched_inputs, batched_targets

    else:
        # inference mode: (input_encoded, prefix, query, docid)
        inputs_list, prefixes, queries, docids = zip(*batch)

        batched_inputs = {}
        for k in inputs_list[0].keys():
            tensors = [inp[k].squeeze(0) for inp in inputs_list]  # remove tokenizers leading dim of 1
            batched_inputs[k] = torch.stack(tensors, dim=0)

        # return lists for the string fields
        return batched_inputs, list(prefixes), list(queries), list(docids)


accelerator = Accelerator()

device = accelerator.device
print("PROCESS STARTED")

if __name__ == '__main__':
    # python src/code/T5/infer.py --inp datasets/master/queries-inference/test_formatted_seen_query-seen_doc_test.tsv --out datasets/results/$MODEL_NAME/$EXP_TYPE/completions_seen_query-unseen_doc_test.mpc --doc datasets/master/docs/trec_test.csv --mdmax_length 48 --input_type $EXP_TYPE --model_name $MODEL_NAME --beam_size 25 --bs $BATCH_SIZE --trie_path $TRIE_PATH --alpha 0.3 --beta 0.1 --bias_strength 30 --use_trie
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--doc', type=str, required=True)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--bs',  type=int, default=4)
    parser.add_argument('--tkmax_length', type=int, default=512)
    parser.add_argument('--mdmax_length', type=int, default=64)
    parser.add_argument("--model_name", type=str, default="t5-small")
    parser.add_argument("--context", action="store_true")
    parser.add_argument("--input_type", type=str, default="full_doc")
    parser.add_argument("--beam_size", type=int, default=25)
    parser.add_argument("--use_trie", action="store_true", help="Use trie for constrained decoding")
    parser.add_argument("--trie_path", type=str, default=None, help="Path to the sequence to be added to the trie")
    parser.add_argument("--alpha", type=float, default=0.3, help="Alpha value for the TrieConstrainedLogitsProcessor")
    parser.add_argument("--beta", type=float, default=0.1, help="Beta value for the TrieConstrainedLogitsProcessor")
    parser.add_argument("--bias_strength", type=float, default=30, help="Bias strength for the TrieConstrainedLogitsProcessor")
    args = parser.parse_args()

    print("Using device:", device)

    # load tokenizer
    print("Loading tokenizer and model...")
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, truncation_side='left', max_length=args.tkmax_length)
    if args.input_type == "yake":
        tokenizer.add_tokens(['<tspace>', "<eok>"])
        tokenizer.add_tokens(["query:", "title:", "context:"])
    else:
        tokenizer.add_tokens('<tspace>')
        tokenizer.add_tokens(["query:", "title:", "context:"])
    model.resize_token_embeddings(len(tokenizer))
    
    
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

    # if ckpt does not end with .pth, then take the latest checkpoint
    if(args.ckpt is not None and not args.ckpt.endswith(".pth")):
        ckpt_files = os.listdir(args.ckpt)
        ckpt_files = [f for f in ckpt_files if f.endswith(".pth")]
        ckpt_files = sorted(ckpt_files)
        if(len(ckpt_files)>0):
            args.ckpt = os.path.join(args.ckpt, ckpt_files[-1])
        else:
            args.ckpt = None

    if(args.ckpt is not None):
        ckpt = torch.load(args.ckpt)
        model.load_state_dict(ckpt["model_state_dict"])
        print("Checkpoint loaded - ", args.ckpt)
    else:
        print("No checkpoint loaded, using vanilla pretrained model")
    model.eval()

    # load data
    print("Loading data...")
    # with open(args.inp, "r") as f:
    #     data = f.read()
    # dataset = data.split("\n")[:-1]
    # infer_data = pd.DataFrame(dataset)
    # sentences = infer_data.values.flatten().tolist()
    print("Preparing data...")
    dataset = AutocompleteDataset(data_path=args.inp, doc_path=args.doc, tkmax_length=args.tkmax_length, tokenizer=tokenizer, infer=True, in_type=args.input_type)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.bs,collate_fn=collate_fn, shuffle=False)
    print("Data prepared")

    model, data_loader = accelerator.prepare(model, data_loader)

    print("Inferencing...")
    with open(args.out, "w") as f:
        start_time = time.time()
        for batch in tqdm.tqdm(data_loader):
            inputs, inp_prefix, inp_query, docid = batch
            # print(inputs["input_ids"].shape)
            # prefix
            # inputs = list(inputs)
            # targets = list(targets)
            # Prepare data
            # encoding = prefix_encoder(tokenizer, inputs, max_length=args.tkmax_length, batch=True)
            # encoding = encoding.to(device)
            # generate outputs with attention masks
            logit_processor = None
            if args.use_trie:
                if isinstance(token_trie, dict):
                    the_trie = [token_trie[d] for d in docid]
                else:
                    the_trie = [token_trie]
                logit_processor = TrieConstrainedLogitsProcessor(
                    the_trie,
                    inp_prefix,
                    tokenizer,
                    inputs["input_ids"].squeeze(1),
                    alpha=args.alpha,
                    beta=args.beta,
                    initial_bias=args.bias_strength
                )
                logit_processor = [logit_processor]
                
            if(hasattr(model, "module")):
                generated_outputs = model.module.generate(
                    input_ids=inputs["input_ids"].squeeze(1), 
                    attention_mask=inputs["attention_mask"].squeeze(1), 
                    num_beams=args.beam_size, 
                    max_new_tokens=args.mdmax_length, 
                    early_stopping = True, 
                    return_dict_in_generate=True, 
                    output_scores=True, 
                    logits_processor=logit_processor,
                    num_return_sequences=args.beam_size
                    )
            else:
                generated_outputs = model.generate(
                    input_ids=inputs["input_ids"].squeeze(1), 
                    attention_mask=inputs["attention_mask"].squeeze(1), 
                    num_beams=args.beam_size, 
                    max_new_tokens=args.mdmax_length, 
                    early_stopping = True, 
                    return_dict_in_generate=True, 
                    output_scores=True, 
                    logits_processor=logit_processor,
                    num_return_sequences=args.beam_size
                    )

            #print the generated sequences
            # print("Generated sequences:")
            # print(tokenizer.batch_decode(generated_outputs.sequences, skip_special_tokens=True))
            # exit(0)

            # generated_outputs = generated_outputs.view(-1, 25, args.mdmax_length)

            gen_sequences = generated_outputs.sequences[:, 1:] # input_length is the length of the input prompt for decoder-only models, like the GPT family, and 1 for # encoder-decoder models, like BART or T5.

            # let's stack the logits generated at each step to a tensor and transform
            # logits to probs
            probs = torch.stack(generated_outputs.scores, dim=1).softmax(-1) # -> shape [3, 15, vocab_size]

            # now we need to collect the probability of the generated token
            # we need to add a dummy dim in the end to make gather work
            # print(probs.shape)
            # print(gen_sequences.shape)
            try:
                gen_probs = torch.gather(probs, 2, gen_sequences[:, :, None]).squeeze(-1)
            except:
                print("Exception occured while calculating probabilities")
                print(gen_sequences)
                print(probs)
                print(tokenizer.batch_decode(gen_sequences, skip_special_tokens=True))
                exit(0)

            # get the average negative log likelihood across generated tokens for each sequence that are not pad tokens

            mask = gen_sequences != tokenizer.pad_token_id
            mask = mask.type(torch.FloatTensor).to(device)

            scores = generated_outputs.sequences_scores
            scores = scores.cpu()
            scores = scores.view(-1, args.beam_size)

            nll = -torch.log(gen_probs) * mask
            nll = nll.sum(1)
            # subword_lens = mask.sum(1)
            gen_sequences = gen_sequences.cpu()
            gen_sequences = gen_sequences.view(-1, args.beam_size, gen_sequences.shape[-1])
            nll = nll.cpu()
            # subword_lens = subword_lens.cpu()
            nll = nll.view(-1, args.beam_size)
            # subword_lens = subword_lens.view(-1, 25)

            seen_comp = set()

            for i in range(inputs["input_ids"].shape[0]):
                dct = {"query": [], "prefix": [], "complitions": []}
                prefix_ = tokenizer.decode(inputs["input_ids"].squeeze(1)[i, :], skip_special_tokens=True)
                # print(prefix)
                query_idx = prefix_.find("query:")
                query = prefix_[query_idx+7:]
                prefix = query.replace("<tspace>", " ")
                if args.input_type == "no_doc" or query_idx < 0:
                    prefix = prefix_.replace("<tspace>", " ")
                # print(len(inputs))
                # gt = targets[i]
                pred_list = []
                confidence_list = []
                for j in range(args.beam_size):
                    pred = suffix_decoder(tokenizer, gen_sequences[i, j])
                    total_sentence = merge_prefix_suffix(prefix, pred)
                    pred = total_sentence[len(prefix):]
                    confidence = str(scores[i, j].item())
                    if total_sentence not in seen_comp:
                        seen_comp.add(pred)
                        pred_list.append([total_sentence, confidence])
                
                out = {"docid": docid[i], "query": inp_query[i], "prefix": inp_prefix[i], "completions": pred_list}

                # f.write(str(out) + "\n")
                f.write(json.dumps(out) + '\n')
                    # subword_len = str(subword_lens[i].item())
                # print(total_sentence)
                # print([prefix, gt, pred, confidence, str(subword_len)])
                # f.write("\t".join([prefix, gt, pred, confidence, str(subword_len)]) + "\n")

        end_time = time.time()

    run_time_file = "datasets/results/t5_run_time.txt"
    # save the run time in a file without removing the previous content
    with open(run_time_file, "a") as f:
        f.write(f"{args.out} run time: {end_time - start_time}s\n")
