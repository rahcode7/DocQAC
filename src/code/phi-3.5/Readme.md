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
>cd phi-3.5
>chmod +x finetune_<model_name>.sh
>./finetune.sh <batch_size>
# example command
>./finetune.sh 8
```

**4.Inference from SLMs**
Go to specific model folder that you've finetuned earlier and run the following 
```
>cd phi-3.5
>chmod +x generate.sh
>./generate.sh
```


Make sure to update the `peft_checkpoint_path` variable in the `generate.sh` script with the path to your finetuned model checkpoint.
