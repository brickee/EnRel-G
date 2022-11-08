# Put the downloaded BioBERT to ./pybiobert_base and SciBERT to ./scibert_scivocab_uncased
# run this shell with the parameters
# If you want to to change the dataset, make sure to change --data_dir and the file names in data_utils.py
# run this file using command: sh run.sh
python run_ner.py --data_dir=data/AnatEM --bert_model=pybiobert_base --task_name=ner --max_seq_length=128 --num_train_epochs=100 --learning_rate=5e-5 --train_batch_size=32 --eval_batch_size=32 --do_train --do_eval --do_predict --seed=42 --use_rnn --use_crf --use_gat --gat_type=AF --fuse_type=v --do_lower_case --relearn_embed --warmup_proportion=0.1