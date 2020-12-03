function train_model()
{
  WORK_DIR=$1
  log_file=$2
  CUDA_VISIBLE_DEVICES=${CUDA_DEVICE_ID} python3 main.py \
    --task_name=${TASK_NAME} \
    --do_train_and_eval=${DO_TRAIN_AND_EVAL} \
    --do_train=${DO_TRAIN} \
    --do_eval=${DO_EVAL} \
    --do_predict=${DO_PREDICT} \
    --data_dir=${DATA_DIR} \
    --vocab_file=${VOCAB_FILE} \
    --bert_config_file=${BERT_CONFIG_FILE} \
    --init_checkpoint=${INIT_CHECKPOINT} \
    --use_crf=${USE_CRF} \
    --use_lstm=${USE_LSTM} \
    --lstm_hidden_dim=${LSTM_HIDDEN_DIM} \
    --num_lstm_layers=${NUM_LSTM_LAYERS} \
    --max_seq_length=${MAX_SEQ_LENGTH} \
    --train_batch_size=${TRAIN_BATCH_SIZE} \
    --predict_batch_size=${PREDICT_BATCH_SIZE} \
    --learning_rate=${LEARNING_RATE} \
    --save_checkpoints_steps=${SAVE_CHECKPOINTS_STEPS} \
    --max_steps_without_increase=${MAX_STEPS_WITHOUT_INCREASE} \
    --num_train_epochs=${NUM_TRAIN_EPOCHS} \
    --do_ensemble=${DO_ENSEMBLE} \
    --ensemble_num=${ENSEMBLE_NUM} \
    --work_dir=${WORK_DIR} \
    >> ${log_file} 2>&1
}
config_file=$1
source ${config_file}

if [ ${DO_ENSEMBLE} == "true" ]; then
  echo "TODO"
else
  log_file=${WORK_DIR}/run.log
  # 生成训练集，测试集
  python3 data_prepare.py ${train_size} >${log_file} 2>&1
  train_model ${WORK_DIR} ${log_file}
  python3 write_result.py ${WORK_DIR}/test_results.tsv >>{log_file} 2>&1
fi