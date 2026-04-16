import pandas as pd
import yake
import argparse
# from transformers import AutoTokenizer

def get_args():    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="t5")

args = get_args()

k_text = ""
split_token = " <eok> "
if args.model == "gpt":
    k_text = "gpt2"
    split_token = " <|eok|> "


language = "en"
max_ngram_size = 3
deduplication_thresold = 0.9
deduplication_algo = 'seqm'
windowSize = 1
numOfKeywords = 50

extractor = yake.KeywordExtractor(
    lan=language, 
    n=max_ngram_size, 
    dedupLim=deduplication_thresold, 
    dedupFunc=deduplication_algo, 
    windowsSize=windowSize, 
    top=numOfKeywords, 
    features=None
    )

def yake_extract(text):
    keywords = extractor.extract_keywords(str(text))
    keywords = [str(k[0]) for k in keywords]
    return {"yake": keywords}

def trunc_text(self, text, max_length):
    text = text.split(" ")
    text = text[:max_length]
    tokenized = self.tokenizer(" ".join(text), max_length=max_length, truncation=True, return_tensors='pt', padding=False)
    text = self.tokenizer.decode(tokenized['input_ids'][0], skip_special_tokens=True)
    return {"text": text}


files = ["datasets/master/summaries/trec_train.csv", "datasets/master/summaries/trec_val.csv", "datasets/master/summaries/trec_test.csv"]

for file in files:
    doc = pd.read_csv(file)
    if ".tsv" in file:
        save_path = file.split(".tsv")[0] + f"_yake{k_text}.tsv"
    else:
        save_path = file.split(".csv")[0] + f"_yake{k_text}.csv"
    
    try:
        if ".tsv" in save_path:
            doc = pd.read_csv(save_path, sep="\t")
        else:
            doc = pd.read_csv(save_path)
    except:

        doc["yake"] = doc["body"].progress_apply(lambda x: yake_extract(x)["yake"])
        doc["yake"] = doc["yake"].apply(lambda x: split_token.join(x))
        
        if ".tsv" in save_path:
            doc.to_csv(save_path, sep="\t", index=False)
        else:
            doc.to_csv(save_path, index=False)
