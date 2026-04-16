# DocQAC: Adaptive Trie-Guided Decoding for Effective In-Document Query Auto-Completion 
### (Accepted as full paper in SIGIR 2026)

Query auto-completion (QAC) has been widely studied in the context of web search, yet remains underexplored for in-document
search, which we term DocQAC. DocQAC aims to enhance search productivity within long documents by helping users craft faster,
more precise queries, even for complex or hard-to-spell terms. Unlike traditional WebQAC systems, DocQAC can leverage rich document context, having access not only to the partially typed user
query and global historical queries, but also the content of the current document itself, and crucially, the document-specific history of user query interactions.

## Dataset 
The DocQAC dataset can be downloaded [here](https://bit.ly/3IGEkbH)




## Reproducibility Guide

###### Dataset 
- Follow code in src/data-prep/master-data to prepare DocQAC benchmark dataset

###### Input Represntations
- Follow code in src/data-prep to create representations for summary,keyphrases and RAG

###### Experiments
- Follow code in src/code for each individual experiments

###### Evaluation 
- Follow code in src/eval for evaluation of each individual experiments
