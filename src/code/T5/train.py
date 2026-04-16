# import required libraries
import pandas as pd
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
# from transformers import AdamW old
from torch.optim import AdamW
import argparse
import os
import wandb
from accelerate import Accelerator
import tqdm
import numpy as np
import namegenerator
import sys
sys.path.append('./code')
from utils import AutocompleteDataset, merge_prefix_suffix, suffix_decoder
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

accelerator = Accelerator()
device = accelerator.device
print("PROCESS STARTED")

def get_args():    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--train_doc', type=str, required=True)
    parser.add_argument('--val_doc', type=str, required=True)
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--num_epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--val_data', type=str, default=None)
    parser.add_argument('--bs',  type=int, default=4)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--tkmax_length', type=int, default=512)
    parser.add_argument('--mdmax_length', type=int, default=512)
    parser.add_argument('--initial_eval', action='store_true')
    parser.add_argument('--eval_every', type=int, default=7500)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--dev', action='store_true')
    parser.add_argument('--model_name', type=str, default="t5-base")
    parser.add_argument('--input_type', type=str, default="full_doc")
    parser.add_argument('--suffix_trie_path', type=str, default="datasets/outputs/global-tries/suffix.mpc")
    parser.add_argument('--main_trie_path', type=str, default="datasets/outputs/global-tries/main.mpc")
    parser.add_argument('--k_comp', type=int, default=10)
    args = parser.parse_args()
    return args

def main(args):
    #come with a new interseting name every time it is none
    if(args.model_dir is None and args.ckpt is None and accelerator.is_main_process):
        args.model_dir = args.model_name + "-" + namegenerator.gen()
        if(args.dev):
            args.model_dir += "-dev"
        os.makedirs(args.model_dir)
        # save args to model directory along with model name
        with open(os.path.join(args.model_dir, "args.txt"), "w") as f:
            f.write(args.model_name + "\n")
            f.write(str(args))
        
        print("The model directory is created!")
        print("Model Directory: ", args.model_dir)
    
    print("Using device:", device)

    # with open(args.train_data, "r") as f:
    #   data = f.read()
    # dataset = data.split("\n")
    # train_data = pd.DataFrame(dataset)
    # sentences = train_data.values.flatten().tolist()

    # Load pre-trained model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, truncation_side='left', tkmax_length=args.tkmax_length)
    if args.input_type == "yake":
        tokenizer.add_tokens(['<tspace>', "<eok>"])
        tokenizer.add_tokens(["query:", "title:", "context:"])
    else:
        tokenizer.add_tokens('<tspace>')
        tokenizer.add_tokens(["query:", "title:", "context:"])
    # if(args.context):
    #     tokenizer.add_tokens('<|EOU|>')
    model.resize_token_embeddings(len(tokenizer))
    optimizer = AdamW(model.parameters(), lr=args.lr)
    # load ckpt if any
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location='cpu')
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        initial_epoch = ckpt["epoch"] + 1
        args.model_dir = os.path.dirname(args.ckpt)
        print("model loaded from checkpoint")
        print("model directory: ", args.model_dir)
    else:
        initial_epoch = 0
        print("pretrained model loaded -- no checkpoint found")


    if(args.wandb and accelerator.is_main_process):
        key = "f45b845032e8ea198753b42b028b7e08701e5ecc"
        wandb.login(key=key, relogin=True)
        wandb.init(project="T5-autocompletion", name=args.model_dir)
    else:
        wandb.init(project="T5-autocompletion", name=args.model_dir, mode="disabled")

    # log to wandb the model directory
    wandb.config.update(args)
    # setup dataloader and optimizer
    print("Tokenizing sentences...")
    dataset = AutocompleteDataset(data_path=args.train_data, doc_path=args.train_doc, tkmax_length=args.tkmax_length, tokenizer=tokenizer, in_type=args.input_type, main_trie_path=args.main_trie_path, suffix_trie_path=args.suffix_trie_path, k_comp=args.k_comp)
    print("total size of train dataset: ", len(dataset))
    dataloader = DataLoader(dataset, batch_size=args.bs, num_workers=4)
    if args.val_data is not None:
        # with open(args.val_data, "r") as f:
        #   data = f.read()
        # dataset = data.split("\n")
        # val_data = pd.DataFrame(dataset)
        # val_sentences = val_data.values.flatten().tolist()
        print("Tokenizing validation sentences...")
        val_dataset = AutocompleteDataset(data_path=args.val_data, doc_path=args.val_doc, tkmax_length=args.tkmax_length, tokenizer=tokenizer, in_type=args.input_type, main_trie_path=args.main_trie_path, suffix_trie_path=args.suffix_trie_path, k_comp=args.k_comp)
        print("total size of validation dataset: ", len(val_dataset))
        val_dataloader = DataLoader(val_dataset, batch_size=args.bs)
        
    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    

    print("STARTING TRAINING")
    model, optimizer, dataloader, val_dataloader = accelerator.prepare(model, optimizer, dataloader, val_dataloader)


    if args.initial_eval:
        model.eval()
        total_val_loss = 0
        print("STARTING INITIAL EVAL")
        with torch.no_grad():
            for batch in tqdm.tqdm(val_dataloader):
                inputs, targets = batch
                # Prepare data
                input_ids = inputs['input_ids'].squeeze(1)
                attention_mask = inputs['attention_mask'].squeeze(1)
                labels = targets['input_ids'].squeeze(1)
                labels[labels == tokenizer.pad_token_id] = -100
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)

        print(f"Initial Evaluation, Validation Loss: {avg_val_loss}")
        # also log perplexity
        wandb.log({
            "avg_val_loss": avg_val_loss,
            "perplexity_val_set": torch.exp(torch.tensor(avg_val_loss))
        })

    
    prev_val_loss = 1000000000
    prev_epoch = -1
    for epoch in tqdm.tqdm(range(initial_epoch, args.num_epochs)):
            # Training phase
            model.train()
            total_train_loss = 0
            iterations = 0
            for batch in tqdm.tqdm(dataloader):
                inputs, targets = batch
                # Prepare data
                input_ids = inputs['input_ids'].squeeze(1)
                attention_mask = inputs['attention_mask'].squeeze(1)
                labels = targets['input_ids'].squeeze(1)
                labels[labels == tokenizer.pad_token_id] = -100
                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                
                # if iterations%100 == 0:
                #     # Decode and print input text
                #     input_text = [tokenizer.decode(input_ids[0], skip_special_tokens=True)]
                #     # find index of "query:" in input_text
                #     # query_idx = input_text[0].find("query:")
                #     # query = input_text[0][query_idx+7:]
                #     query = input_text[0]
                #     # print(f"Input Text (Epoch: {epoch}, Iteration {iterations}): {query}")
                    
                #     # Generate model output and decode
                #     with torch.no_grad():
                #         # see if model has attribute modules
                #         if hasattr(model, "module"):
                #             model_output = model.module.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=args.mdmax_length)
                #         else:
                #             model_output = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=args.mdmax_length)
                #         output_text = suffix_decoder(tokenizer, model_output[0].detach().clone().cpu())
                #         query = query.replace("<tspace>", " ")
                #         output_text = [merge_prefix_suffix(query, output_text)]

                #         gt_enc = labels[0].detach().clone().cpu()
                #         gt_enc[gt_enc == -100] = tokenizer.pad_token_id
                #         gt_text = [merge_prefix_suffix(query, suffix_decoder(tokenizer, gt_enc))]
                        
                #         # print(f"Model Output (Epoch: {epoch}, Iteration {iterations}): {output_text}, Ground Truth: {gt_text}")
                    
                #     wandb.log({
                #         "training loss": loss.item(),
                #         })
                
                total_train_loss += loss.item()
                # Evaluation phase
                if(args.val_data is not None and (iterations+1)%args.eval_every == 0):
                    model.eval()
                    total_val_loss = 0
                    print("STARTING VALIDATION")
                    with torch.no_grad():
                        for batch in tqdm.tqdm(val_dataloader):
                            inputs, targets = batch
                            # Prepare data
                            input_ids = inputs['input_ids'].squeeze(1)
                            attention_mask = inputs['attention_mask'].squeeze(1)
                            labels = targets['input_ids'].squeeze(1)
                            labels[labels == tokenizer.pad_token_id] = -100
                            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                            loss = outputs.loss

                            total_val_loss += loss.item()

                    avg_val_loss = total_val_loss / len(val_dataloader)

                    print(f"Epoch: {epoch}, Iteration: {iterations}, Training Loss: {loss.item()}, Validation Loss: {avg_val_loss}")
                    # also log perplexity
                    wandb.log({
                        "avg_val_loss": avg_val_loss,
                        "perplexity_val_set": torch.exp(torch.tensor(avg_val_loss))
                    })
                    
                    model.train()

                iterations = iterations + 1
            
            avg_train_loss = total_train_loss / len(dataloader)

            if args.val_data is not None:
                model.eval()
                total_val_loss = 0
                print("STARTING VALIDATION AT END OF EPOCH")
                with torch.no_grad():
                    for batch in tqdm.tqdm(val_dataloader):
                        inputs, targets = batch
                        # Prepare data
                        input_ids = inputs['input_ids'].squeeze(1)
                        attention_mask = inputs['attention_mask'].squeeze(1)
                        labels = targets['input_ids'].squeeze(1)
                        labels[labels == tokenizer.pad_token_id] = -100
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        loss = outputs.loss

                        total_val_loss += loss.item()

                avg_val_loss = total_val_loss / len(val_dataloader)

                print(f"Epoch: {epoch}, Training Loss: {loss.item()}, Validation Loss: {avg_val_loss}")
                # also log perplexity
                wandb.log({
                    "avg_val_loss": avg_val_loss,
                    "perplexity_val_set": torch.exp(torch.tensor(avg_val_loss)),
                    "epoch": epoch
                })
                
                model.train()

            if hasattr(model, "module"):
                _model = accelerator.unwrap_model(model.module)
                # _optimizer = optimizer.module
            else:
                _model = model
                # _optimizer = optimizer

            print(f"Epoch: {epoch}, Training Loss: {avg_train_loss}")
            wandb.log({
                "avg_train_loss": avg_train_loss
                })
            # Save model checkpoint at the end of each epoch only if is the main process
            if(accelerator.is_main_process) and prev_val_loss > avg_val_loss:
                prev_val_loss = avg_val_loss
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': _model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_train_loss,
                }
                checkpoint_path = os.path.join(args.model_dir, f'epoch_{epoch}.pth')
                torch.save(checkpoint, checkpoint_path)
                print(f"Checkpoint saved at '{checkpoint_path}'")

                # print(os.path.join(args.model_dir, f'epoch_{prev_epoch}.pth'))

                if prev_epoch != -1 and os.path.exists(os.path.join(args.model_dir, f'epoch_{prev_epoch}.pth')) and prev_epoch != epoch:
                    os.remove(os.path.join(args.model_dir, f'epoch_{prev_epoch}.pth'))
                    print(f"Checkpoint deleted at 'epoch_{prev_epoch}.pth'")

                prev_epoch = epoch


if __name__ == "__main__":
    main(get_args())
