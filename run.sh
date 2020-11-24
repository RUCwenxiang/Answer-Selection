export BERT_BASE_DIR=./chinese_L-12_H-768_A-12
export MY_DATA_DIR=./data/
python3 main.py \
  --task_name=answer_sent_labeling \
  --do_train=true \
  --do_eval=true \
  --data_dir=$MY_DATA_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=3 \
  --output_dir=./checkpoint_answer_sent_labeling