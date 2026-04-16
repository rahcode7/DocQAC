from datasets import load_dataset
from argparse import ArgumentParser
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)

import math
import torch
from task_utils import prepare_input, valid_record
import sys
from functools import partial 
import os
import random

torch.random.manual_seed(0)

exp_variations = [
    "prefix_only",
    "title_doc",
    "title_url",
    "title_url_doc",
    "title_url_summary",
    "title_url_yake",
    "title_url_rag_sparse",
    "title_url_rag_dense",
    "title_url_rag_sim_doc",
]

args = ArgumentParser()
args.add_argument("--data_path", type=str, required=True)
args.add_argument("--checkpoint_path", type=str, required=True)
args.add_argument("--max_seq_len", type=int, default=512)
args.add_argument("--epochs", type=int, default=1)
args.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
args.add_argument("--batch_size", type=int, default=8)
args.add_argument("--grad_accum", type=int, default=1)
args.add_argument("--train_data_count", type=int, default=-1)
args.add_argument("--val_data_count", type=int, default=-1)
args.add_argument("--context_type", type=str, choices=exp_variations, default="prefix_only")
args = args.parse_args()

IGNORE_INDEX=-100
MAX_SEQ_LEN=args.max_seq_len

# storage paths
os.makedirs(args.checkpoint_path, exist_ok=True)

model_id=args.model_name
dataset = load_dataset(args.data_path, data_files={'train': 'train.tsv', 'val': 'val.tsv'})

# check dataset validity : removes empty records
dataset = dataset.filter(lambda x: valid_record(x))

# loading the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, add_eos_token=True)
tokenizer.pad_token = "<|reserved_special_token_0|>"  # use unk rather than eos token to prevent endless generation
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
tokenizer.padding_side = 'right'

# loading the model
model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        trust_remote_code=True,
        use_flash_attention_2=False)
model.config.eos_token_id = tokenizer.eos_token_id

#parameter efficient fine-tuning
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
model.config.use_cache = False
model.config.pretraining_tp = 1

# train on small sample of 10000 instances
if args.train_data_count != -1:
    print(f'reducing train count from {len(dataset["train"])} to {args.train_data_count}')
    dataset["train"] = dataset["train"].select(range(args.train_data_count))

if args.val_data_count != -1:
    print(f'reducing val count from {len(dataset["val"])} to {args.val_data_count}')
    dataset["val"] = dataset["val"].select(range(args.val_data_count))

def tokenize(input, max_length):
    chat_template = f"{tokenizer.apply_chat_template(prepare_input(tokenizer, input, args.context_type)[0], tokenize=False, add_generation_prompt=False)}{tokenizer.eos_token}"
    return_dict = tokenizer.encode_plus(chat_template, truncation=True, 
                                        max_length=max_length, 
                                        pad_to_max_length=False, 
                                        return_attention_mask=True)
    return_dict.update({"labels": return_dict["input_ids"]})
    return return_dict

dataset_tokenized = dataset.map(
    partial(tokenize, max_length=MAX_SEQ_LEN),
    batched = False,
    num_proc = os.cpu_count()//torch.cuda.device_count(), # parallel threading
    remove_columns = dataset["train"].column_names
)

def collate(elements):
    # Extract input_ids from each element and find the maximum length among them 
    tokens = [e["input_ids"] for e in elements]  
    tokens_maxlen = max([len(t) for t in tokens])  
  
    for i, sample in enumerate(elements):  
        input_ids = sample["input_ids"]  
        labels = sample["labels"]  
        attention_mask = sample["attention_mask"]  
  
        # Calculate the padding length required to match the maximum token length  
        pad_len = tokens_maxlen-len(input_ids)  
  
        # Pad 'input_ids' with the pad token ID, 'labels' with IGNORE_INDEX, and 'attention_mask' with 0  
        input_ids.extend( pad_len * [tokenizer.pad_token_id] )  
        labels.extend( pad_len * [IGNORE_INDEX] )  
        attention_mask.extend( pad_len * [0] )  
  
    # create and return batch with all the data in elements  
    batch={  
        "input_ids": torch.tensor( [e["input_ids"] for e in elements] ),  
        "labels": torch.tensor( [e["labels"] for e in elements] ),  
        "attention_mask": torch.tensor( [e["attention_mask"] for e in elements] ),  
    }  
    return batch

steps_per_epoch=math.ceil(len(dataset["train"])/(args.batch_size*args.grad_accum*torch.cuda.device_count()))
print("steps per epoch: ", steps_per_epoch)

training_arguments = TrainingArguments(
    output_dir=args.checkpoint_path,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=2*args.batch_size,
    gradient_accumulation_steps=args.grad_accum,
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=steps_per_epoch//2,
    logging_steps=steps_per_epoch//4,
    save_steps=steps_per_epoch,
    optim="paged_adamw_32bit",
    num_train_epochs=args.epochs,
    log_level="debug",
    fsdp="full_shard",
    learning_rate=5e-5,
    lr_scheduler_type="constant",
    weight_decay=0.01,
    fp16=True,
    group_by_length=False, 
    ddp_find_unused_parameters=False,
)

trainer = Trainer(
    model=model,
    train_dataset=dataset_tokenized["train"],
    eval_dataset=dataset_tokenized["val"],
    tokenizer=tokenizer,
    data_collator=collate,
    args=training_arguments,
)

trainer.train()
