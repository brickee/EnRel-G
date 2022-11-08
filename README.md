# EnRel-G

A Lightweight System for Improving the domain-specific Named Entity Recognition 

## Training & Prediction
1. Put the downloaded [BioBERT](https://github.com/dmis-lab/biobert) to *./pybiobert_base* and [SciBERT](https://github.com/allenai/scibert) to *./scibert_scivocab_uncased* 
2. Put the conll-style data to *./data*. If you want to to change the dataset, make sure to change the parameter *--data_dir* and the file names in *data_utils.py*
3. run the following commands or run the shell file: *sh run.sh*

```

python run_ner.py --data_dir=data/AnatEM --bert_model=pybiobert_base --task_name=ner --max_seq_length=128 --num_train_epochs=100 --learning_rate=5e-5 --train_batch_size=32 --eval_batch_size=32 --do_train --do_eval --do_predict --seed=42 --use_rnn --use_crf --use_gat --gat_type=AF --fuse_type=v --do_lower_case --relearn_embed --warmup_proportion=0.1

```

## Citation

If you use this system, please cite the paper where it was introduced.

[paper link](https://aclanthology.org/2021.acl-short.93.pdf) 
```text
@inproceedings{chen-etal-2021-explicitly,
 author = {Chen, Pei  and Ding, Haibo  and Araki, Jun  and Huang, Ruihong},
 booktitle = {ACL-2021},
 publisher = {Association for Computational Linguistics},
 title = {Explicitly Capturing Relations between Entity Mentions via Graph Neural Networks for Domain-specific Named Entity Recognition},
 url = {https://aclanthology.org/2021.acl-short.93},
 year = {2021}
}
```

