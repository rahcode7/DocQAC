


### Build Document Indexes 

1. Get MSMARCO documents from the train,test,dev set 11-orcas-master.ipynb
2. Create indexes 


```
 python3 00-index-docs.py
```


3. Run full match
Single Core
```
 python3 01-fullmatch.py
```

Multiprocessing enabled
```
python3 01-fullmatch-multip.py
```