# EnRel-G

A Lightweight System for Improving the domain-specific Named Entity Recognition 

## Training & Prediction
1. Put the downloaded BioBERT to *./pybiobert_base* and SciBERT to *./scibert_scivocab_uncased* 
2. Put the conll-style data to *./data*. If you want to to change the dataset, make sure to change the parameter *--data_dir* and the file names in *data_utils.py*
3. run the following commands or run the shell file: *sh run.sh*

```

CUDA_VISIBLE_DEVICES=0 python -W ignore run_ner.py --data_dir=data/AnatEM --bert_model=pybiobert_base --task_name=ner --max_seq_length=128 --num_train_epochs=100 --learning_rate=5e-5 --train_batch_size=32 --eval_batch_size=32 --do_train --do_eval --do_predict --seed=42 --use_rnn --use_crf --use_gat --gat_type=AF --fuse_type=v --do_lower_case --relearn_embed --warmup_proportion=0.1

```

## Citation

If you use this system, please cite the paper (to appear in ACL 2021) where it was introduced.

[paper]() 
```text
@inproceedings{chen-etal-2021-probing,
 author = {Chen, Pei  and Liu, Kang  and Chen, Yubo  and Wang, Taifeng  and Zhao, Jun},
 booktitle = {Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume},
 month = {April},
 pages = {2042--2048},
 publisher = {Association for Computational Linguistics},
 title = {Probing into the Root: A Dataset for Reason Extraction of Structural Events from Financial Documents},
 year = {2021}
}
```