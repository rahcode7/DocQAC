**1.Conda Setup**<br>
```
>conda create --name slm python=3.9 -y
>conda activate slm
>pip install -r requirements.txt
```
please check conda [installation](https://docs.conda.io/en/latest/miniconda.html) if you don't have conda installed.

**2.Preparing training data for RAG**
For RAG based training, we first prepare the chunks and concatenate them which can directly fed to model (similar to document content).

Go to the `slm_prepare_data_for_rag` and run the following command:
```
>cd ../slm_prepare_data_for_rag
>chmod +x prep_data.sh
>./prep_data.sh
```

In the above, we create different columns in the data frame for different rag based techniques:
- rag_sparse : BM25 retrieval
- rag_dense : Embedding based retrieval (we use SentenceBert for encoding)
- rag_sim_doc : Embedding based retrieval span across related documents. 

**3.Finetune SLMs**
Go to specific model folder wish to train and run the following command:
```
>cd llama-3.2
>chmod +x finetune.sh
>./finetune.sh <batch_size>
# example command
>./finetune.sh 8
```

### Additional steps for running the Llama
If you are finetuning the Llama-3.2 model, please upgrade the transformers package after creating the environment as follows:
```
pip install --upgrade transformers==4.46.0
```

Also Llama3 are gated model you need to request for access then export the HF token before running the finetuning/inference scripts as follows:
```
export HF_TOKEN=<your_key>
```

**4.Inference from SLMs**
Go to specific model folder that you've finetuned earlier and run the following 
```
>cd llama-3.2
>chmod +x generate.sh
>./generate.sh
```
Make sure to update the `peft_checkpoint_path` variable in the `generate.sh` script with the path to your finetuned model checkpoint.


