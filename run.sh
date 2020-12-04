function train_model()
{
  work_dir=$1
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
    --work_dir=${work_dir} \
    >> ${log_file} 2>&1
}

function pipeline()
{
  work_dir=$1
  do_ensemble=$2
  log_file=$3

  # 生成文件路径
  mkdir -p ${work_dir}

  # 训练模型
  train_model ${work_dir} ${log_file}
  logging $? ${log_file} "ERROR: train model meets error!" "LOG: train model successfully!"

  # 写结果
  if [ ${do_ensemble} == "false" ]; then
    python3 -c "from utils import result_file_to_submission_file;
      result_file_to_submission_file(output_predict_file=${work_dir}/test_results.tsv)" >>${log_file} 2>&1
    logging $? ${log_file} "ERROR: write result meets error!" "LOG: write result successfully!"
  fi
}

function logging()
{
  error=$1
  log_file=$2
  error_info=$3
  correct_info=$4
  if [ ${error} -gt 0 ]; then
    echo ${error_info} >>${log_file}
  fi
  echo ${correct_info} >>${log_file}
}

function data_prepare()
{
  train_size=$1
  log_file=$2
  # 生成训练集，测试集
  python3 data_prepare.py ${train_size} >${log_file} 2>&1
  logging $? ${log_file} "ERROR: data prepare meets error!" "LOG: data prepare successfully!"
}
config_file=$1
source ${config_file}

if [ ${DO_ENSEMBLE} == "true" ]; then
  log_file=ensemble_run.log
  data_prepare ${TRAIN_SIZE} ${log_file}
  for((i=1;i<ENSEMBLE_NUM;i++)); do
    work_dir=${WORK_DIR}-$i
    pipeline ${work_dir} ${DO_ENSEMBLE} ${log_file}
  done
  python3 -c "from utils import ensemble_results_to_submission_file;
    ensemble_results_to_submission_file(model_name=${WORK_DIR}, model_num=${ENSEMBLE_NUM})" >>${log_file} 2>&1
  logging $? ${log_file} "ERROR: write ensemble model's results meets error!" "LOG: write ensemble model's results successfully!"
else
  log_file=${WORK_DIR}/run.log
  data_prepare ${TRAIN_SIZE} ${log_file}
  pipeline ${WORK_DIR} ${DO_ENSEMBLE} ${log_file}
fi
# 上传结果
obsutil cp submission.tsv obs://sprs-data-sg/NeverDelete/wenxiang/misc/