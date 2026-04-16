import pandas as pd 
import numpy as np 
import os
from tqdm import tqdm 
import os
from openai import AzureOpenAI
import seaborn as sns
import matplotlib.pyplot as plt 
tqdm.pandas()

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def generate_summary(input_text):
    client = AzureOpenAI(
    api_key = os.getenv("OPENAI_API_KEY"),  
    api_version = "2024-02-01",
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    )

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME, # model = "deployment_name".
            messages=[
              
            #{"role": "system", "content": "Assistant is a large language model trained by OpenAI. Summarize the issues faced by the user step by step."}, # Chain of thought prompting 
            #{"role": "system", "content": "Assistant is a large language model trained by OpenAI. Summarize the given document"},
            {"role": "system", "content": "As a document summarizer, summarize the main points of the given article in 300 words."},
            #{"role": "system", "content": "As a document summarizer,summarize the main points of this article in 300 words. You are given the heading and content of the document"},
            #{"role": "system", "content": "Assistant is a large language model trained by OpenAI. Summarize the issues faced by the user in 400 words"},
            #{"role": "system", "content": "Assistant is a large language model trained by OpenAI. Summarize the given document"},
            {"role": "user","content": f"{input_text}"}
            ],
            stream = False
        )
        res = completion.choices[0].message.content
        #print(res) 
        
        return res 
    except:
        res = ""  
        print("no results") 
        return res 
    
def trim_docs(doc):
    doc = str(doc)
    d = doc.split(" ")
    d = d[0:10500]
    return " ".join(d)


if __name__ == "__main__":

    os.environ['OPENAI_API_KEY'] = 'YOUR_API_KEY'
    os.environ['AZURE_OPENAI_ENDPOINT'] = "YOUR_AZURE_INSTANCE"
    os.environ['CHAT_COMPLETIONS_DEPLOYMENT_NAME'] = "gpt-35-turbu-16k-x"
    MODEL_NAME = "gpt-35-turbu-16k-x"
    ENDPOINT = "https://x-gpt-instance.openai.azure.com"


    # Read dataframe 
    DATA_PATH = "datasets/master"
    #df = pd.read_csv(os.path.join(DATA_PATH,"trec_train.csv"),on_bad_lines='skip')  # train
    #df = pd.read_csv(os.path.join(DATA_PATH,"trec_val.csv"),on_bad_lines='skip')  # val
    df = pd.read_csv(os.path.join(DATA_PATH,"trec_test.csv"),on_bad_lines='skip')  # test
     

    print(f'Number of docs',df.shape[0])
    #print(df.head(3))
    df = df.dropna(subset=['body'])
    print(df.shape) 

    
    # Get length of documents 
    #df =  df.head(10)
    print(df['body_length'].max(),df['body_length'].min())
    # sns.histplot(data=df, x="body_length")
    # plt.show()
    # Get max words 

    print(df.columns)
    # if body_length > 11000 words, trim it 

    # Add heading + body if required
    #df["body"] =  "Heading: " + df["heading"] + " Body:" + df["body"]
    df["body"] = df['body'].astype(str)
    #print(df.head(3))

    # Trim documents
    df['body'] = df.progress_apply(lambda x :  trim_docs(x.body if x.body_length > 10500 else x.body),axis=1)
    df['body_length'] = df.progress_apply(lambda x : len(x.body.split()),axis=1)
    print("Max doc words : ",df['body_length'].max(),df['body_length'].min())

    #  Generate Summaries - samples
    #df = df.head(5)

    print("********* Generating summaries via GPT .... ")
    # df['body_summary']  = df.progress_apply(lambda s : generate_summary(s.body),axis=1)
    
    # # Backfill if empty
    # df["body_summary"] = df.progress_apply(lambda x : x.body_summary if x.body_summary else x.body,axis=1)

    # #print(df.columns)
    # df = df[["docid","body_summary","heading"]]
    # df.columns = ['docid','body','heading']
    # #df.to_csv(os.path.join(DATA_PATH,"summaries/trec_train_summary_300words.csv"),index=None)
    # #df.to_csv(os.path.join(DATA_PATH,"summaries/trec_val_summary_300words.csv"),index=None)
    # df.to_csv(os.path.join(DATA_PATH,"summaries/trec_test_summary_300words.csv"),index=None)
    


    # Generate Summaries - samples
    sample_df = df.head(5)
    print(sample_df)
    sample_df['body_summary']  = sample_df['body'].progress_apply(lambda s : generate_summary(s))
    print(sample_df)
    

    for index,row in sample_df.iterrows():
        print("*****************************")
        print(row['body'])
        print("Article length",row['body_length'])
        print("Summary:")
        print(row['body_summary'])

    #sample_df.to_csv(os.path.join(DATA_PATH,"summaries/trec_train_sample_summary_300words_heading.csv"),index=None)
