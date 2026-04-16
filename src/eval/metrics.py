import json 
import os
import pandas as pd 
import numpy as np
import nltk.translate.bleu_score as bleu_score
from transformers import BertTokenizer, BertForMaskedLM, BertModel
from bert_score import BERTScorer
import  nltk #import nltk.translate.bleu_score.sentence_bleu as sentence_bleu
#from nltk #import nltk.translate.bleu_score.SmoothingFunction as SmoothingFunction

########################################################### MEAN RECIPROCAL RANK

def mrr_helper(gt,pred):

    if not pred:
        r = [0] 
    elif not pred[0] and len(pred)==1:
        r = [0]
    else:
        r = []
        for p in pred:
            if gt == p[0]:
                r.append(1)
            else:
                r.append(0)
    return r


def mean_reciprocal_rank(rs):
    """Score is reciprocal of the rank of the first relevant item."""
    rs = (np.atleast_1d(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])


# def mean_reciprocal_rank(rs):
#     """Score is reciprocal of the rank of the first relevant item
#     First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
#     Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
#     >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
#     >>> mean_reciprocal_rank(rs)
#     0.61111111111111105
#     >>> rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
#     >>> mean_reciprocal_rank(rs)
#     0.5
#     >>> rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
#     >>> mean_reciprocal_rank(rs)
#     0.75
#     Args:
#         rs: Iterator of relevance scores (list or numpy) in rank order
#             (first element is the first item)
#     Returns:
#         Mean reciprocal rank
#     """
#     rs = (np.asarray(r).nonzero()[0] for r in rs)
#     return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])


# gt = "hello"
# pred = [["hello","world"],["world","h"]]
# mrr_helper(gt,pred)


# rs = [[0, 0, 1,1,1]] # , [0, 1, 0], [1, 0, 0]]
# mean_reciprocal_rank(rs)

########################################################### NDCG - HELPERS
def partial_precision(gt,p):
    
    # character overlap between gt and pred / len(pred)
    over = 0
    for i,c in enumerate(gt):
        if i >= len(p) or c!=p[i]:  #will be required if 
            break
        if c == p[i]:
            #print(c,p[i])
            over+=1  
    #print(len(p),p)
    if (len(p)==0):
        return 0.0
    else:
        pprec = round(over/len(p),3)
    #print(f'Partial precison {pprec}')
    return pprec

def partial_recall(gt,p):
    gt = gt.replace(" ","")
    p = p.replace(" ","")
    # character overlap between gt and pred / gt
    over = 0
    for i,c in enumerate(gt):
        if i >= len(p) or c!=p[i]:  #will be required if 
            break
        if c == p[i]:
            #print(c,p[i])
            over+=1  

    prec = round(over/len(gt),3)
    #print(len(gt),over)
    return prec



########################################################### NDCG CORE 




def dcg_at_k(r, k):
    #print(r,k)
    #r = np.asfarray(r)[:k]
    r = r[:k]
    #print(r)
    if r:
        #print(np.log2(np.arange(2, r.size + 1)))
        #print(np.sum(r[1:] / np.log2(np.arange(2, r.size + 1))),r[0])
        return r[0] + np.sum(r[1:] / np.log2(np.arange(2, len(r) + 1)))
        #res = round(r[0] + np.sum(r[1:]/np.log2(np.arange(2, len(r) + 1)+1)),3)
        #print(res,res.dtype,float(res),round(res,3))
        #return res

    return round(0.0,3)

def ndcg_at_k(r, k):
    #print(r,k)
    # Ideal NDCG 
    # W.r.t predictions - dont' use
    #dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    # All 1 for binary relevance 
    dcg_max = dcg_at_k([1.0]*len(r), k)
    #print(dcg_max)
    #print(f'dcg max:{dcg_max}')
    # if not dcg_max:
    #     return float(0.0)

    #print(f'dcg {dcg_at_k(r, k)} and dcgmax {dcg_max} and max arr {sorted(r, reverse=True)}')
    if dcg_at_k(r, k)!=0:
        #print(f"{dcg_at_k(r, k)}")
        ndcg = float(dcg_at_k(r, k) / dcg_max)
    else:
        return float(0.0)
    return ndcg

# actual = {'r1':1,'r2':0.5,'r3':0.5}
# recommended = ['r1','r2','r2','r2','r2'] #,'r2','r1','r2','r1','r2']

# Convert recommendation list to relevance scores
# r = [actual[i] for i in recommended]

# print(r,sorted(r, reverse=True))
# print("NDCG@2: ", ndcg_at_k(r, 2))
# print("NDCG@5: ", ndcg_at_k(r, 5))


# In[43]:

########################################################### NDCG - PARTIAL PRECISION


def ndcg_partial_prec(gt,pred,k=10):
    #print(pred)
    if not pred:
        r = [0.0] 
    elif not pred[0] and len(pred)==1:
        r = [0.0]
    else:
        r = []
        for p in pred:
            prec = partial_precision(gt,p[0])
            r.append(prec)

    #print(f'Reco pprec {r}')
    ndcg = ndcg_at_k(r,k)
    #print(f'ndcg {ndcg}')
    
    return ndcg 
# gt =  'endangered animal'
# pred_list =  [['endangered species', 'DQT:1000'], ['endangered animals', 'DQT:694'], ['endangered species list', 'DQT:235'], ['endangered animals list', 'DQT:90'], ['endangered animal', 'DQT:23'], ['endangered animal list', 'DQT:21'], ['endangered species of animals', 'DQT:7'], ['endangered species animals', 'DQT:7'], ['endangered status', 'DQT:4'], ['endangered and extinct animals', 'DQT:2']]
# ndcg_partial_prec(gt,pred_list,k=10)


########################################################### NDCG - PARTIAL RECALL

def ndcg_partial_rec(gt,pred,k=10):
    if not pred:
        r = [0.0] 
    elif not pred[0] and len(pred)==1:
        r = [0.0]
    else:
        # if gt matches pred 
        r = []
        for p in pred:
            prec = partial_recall(gt,p[0])
            r.append(prec)

    #print(f'Reco prec {r}')
    ndcg = ndcg_at_k(r,k)
    #print(f'ndcg {ndcg}')
    return ndcg

########################################################### TES
def is_nested_list_empty(lst):
    return all(not sublist for sublist in lst)

def tes(q,pred_list):
    #print(q,pred_list)

    # Empty query
    if not q:
        return 0.0 
    
    if not pred_list:
        return 0.0
    elif not pred_list[0] and len(pred_list)==1:
        return 0.0
    else:
        total_saved =0 
        i = 0 
        while i < len(q)-1:
            max_saved = 0 
            
            #print(i,max_saved,total_saved,len(pred_list))
            # if there are less predictions than current place

            if i>len(pred_list)-1:
                break

            # Keep top 10 predictions only 
            if len(pred_list[i])>10:
                pred_list[i] = pred_list[i][0:9]

            # if not pred_list[0]:
            #     char_saved = 0
            #else:
                # new code
            for pred in pred_list[i]:
                # Can be single list 
                if not pred:
                    char_saved = 0 
                    continue
                # print(pred)
                #l = len(pred)
                if q.find(pred[0])!=-1 and q.find(pred[0])>=0: 
                    char_saved = len(pred[0]) - (i+1)
                    max_saved = max(max_saved,char_saved) 
                    #print(pred,q,char_saved,max_saved)

            if max_saved == 0:
                i = i +1
            else:
                total_saved += max_saved
                i = max_saved + (i) # +1)
        
        tes_score = total_saved/len(q)
        #print(q,tes_score,total_saved,len(q))
    return tes_score    

# q = "moh m r"
# p = [['moh m'],['m','m'],['',''],['',''],['','']]

# q = "netfl"
# # p = [['net','n','ne'],['','',''],['netfl','',''],['netfl','net','ne'],['','','']]
# print(p[0])
# print(tes(q,p))


# q = "netfl"
# p = [['net','n','ne'],['','',''],['netflix','',''],['netfl','net','ne'],['','','']]
# print(p[0])
# print(tes(q,p))
# # 3

# q = "netfl"
# p = [['net','n','ne'],['','',''],['netflix','',''],['net','net','ne'],['netfl','','']]
# print(p[0])
# print(tes(q,p))
# # 2


# q = "netfl"
# p = [['n','n','n'],['','',''],['n','',''],['n','n','n'],['netfl','','']]
# print(p[0])
# print(tes(q,p))
# 0


# q = "netflix india"
# p = [['netflix india login','n','n'],['','',''],['n','',''],['n','n','n'],['netflix','','']]
# print(p[0])
# print(tes(q,p))
# # # 0

# 1 - 1/5 = 4/5

# 9/10 # 90%

# 1/100 # 0.99 

########################################################### XC - MERGING LOGIC
def get_full_suggestions(prefix, suggestion):
    if prefix[-1] == " ":
        return prefix + suggestion
    idx = prefix.rfind(" ")
    return prefix[:idx+1] + suggestion

########################################################### BLEU Reciprocal Rank
# 4 grams + smoothing

chencherry = nltk.translate.bleu_score.SmoothingFunction()
def bleu_score(ref,hyp):
    ref = ref.split(" ")
    hyp = hyp.split(" ")
    #print(ref,hyp)
    #bleu = round(nltk.translate.bleu_score.sentence_bleu([ref],hyp),4)
    bleu = round(nltk.translate.bleu_score.sentence_bleu([ref],hyp ,smoothing_function=chencherry.method4),4)
    return bleu

def bleu_rr(q,pred_list):
    if not pred_list:
        return 0.0
    elif not pred_list[0] and len(pred_list)==1:
        return 0.0
    else:
        bleu_q = []
        bleu_q_den = []
        for i,p in enumerate(pred_list):
            
            bleu_prefix = bleu_score(q,p[0])/(i+1)
            #print(i,p[0],bleu_prefix)
            bleu_q.append(bleu_prefix)
            bleu_q_den.append(1/(i+1))
        #print(bleu_q,bleu_q_den)

        if not bleu_q or not bleu_q_den:
            return 0.0 
        else:
            bleu_rr = sum(bleu_q) / sum(bleu_q_den)
            #print(bleu_rr)
        #print(bleu_rr)
    return bleu_rr 
    
# q = "netflix india login new"
# p = [['netflix india login new','1'],['netflix','2']]
# p = [['new','1'],['netflix','2']]
# p = [['netflix','2'],['netflix india login new','1'],]
# bleu_rr(q,p)

######################################### Alpha NDCG 
### Set alpha = 0.5
### Zn - normalization constant 

def ndcg_at_k(r, k):
    # Ideal NDCG 
    # W.r.t predictions - dont' use
    #dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    # All 1 for binary relevance 
    dcg_max = dcg_at_k([1.0]*len(r), k)
    #print(dcg_max)
    #print(f'dcg max:{dcg_max}')
    # if not dcg_max:
    #     return float(0.0)

    #print(f'dcg {dcg_at_k(r, k)} and dcgmax {dcg_max} and max arr {sorted(r, reverse=True)}')
    if dcg_at_k(r, k)!=0:
        #print(f"{dcg_at_k(r, k)}")
        ndcg = float(dcg_at_k(r, k) / dcg_max)
    else:
        return float(0.0)
    return ndcg

# Utilize alpha and udpate relevance (1-alpha)*1
def ndcg_alpha(gt,pred,k=10,alpha=0.5):
    if not pred:
        r = [0.0] 
    elif not pred[0] and len(pred)==1:
        r = [0.0]
    else:
        # if gt matches pred 
        r = []
        for p in pred:
            if gt == p[0]:
                r.append(1*(1-alpha))
            else:
                r.append(0)
    #print(f'Reco {r}')
    ndcg = ndcg_at_k(r,k)
    return ndcg 


########################################################### Semantic Score - BERT SCORE Reciprocal Rank 
scorer = BERTScorer(model_type='bert-base-uncased')
def bert_score_rr(gt,pred,k=10):
    bs_prec,bs_rec,bs_f1 = [],[],[]
    for p in pred:
        P, R, F1 = scorer.score([p[0]], [gt])
        #print(P,R,F1,p[0],gt)
        bs_prec.append(P.item())
        bs_rec.append(R.item())
        bs_f1.append(F1.item())


    if not bs_prec or not bs_rec or not bs_f1:
        return 0.0,0.0,0.0  
    else:
        bs_prec_rr = sum(bs_prec)/len(bs_prec)
        bs_rec_rr = sum(bs_rec)/len(bs_rec)
        bs_f1_rr = sum(bs_f1)/len(bs_f1)

    return bs_prec_rr,bs_rec_rr,bs_f1_rr 

        

def semantic_score_helper(model,query,pred_list):
    if not pred_list:
        return 0.0 
    elif not pred_list[0] and len(pred_list)==1:
        return 0.0
    else:
        embed_query = model.encode(query)
        if pred_list:
            pred_list = [p[0] for p in pred_list]
            embed_pred = model.encode(pred_list)
            similarities = model.similarity(embed_query, embed_pred).tolist()[0]
            r = [1 if score >=0.90 else 0.0 for score in similarities]
            #print(query,pred_list,r,similarities)
        else:
            return []
        return r 
