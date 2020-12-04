# 预训练模型的相关路径
BERT_DIR=./chinese_roberta_wwm_ext_base

VOCAB_FILE=${BERT_DIR}/vocab.txt

BERT_CONFIG_FILE=${BERT_DIR}/bert_config.json

INIT_CHECKPOINT=${BERT_DIR}/bert_model.ckpt

# 训练集, 开发集, 测试集的目录
DATA_DIR=./data/

# 任务名, 固定的不需要修改
TASK_NAME=answer_sent_labeling

# 训练评估预测相关
DO_TRAIN_AND_EVAL=false # 模型选点的时候会需要开开

DO_TRAIN=false

DO_EVAL=false

DO_PREDICT=true

# GPU
CUDA_DEVICE_ID=3

# 网络参数
USE_LSTM=true

LSTM_HIDDEN_DIM=256

NUM_LSTM_LAYERS=1

USE_CRF=false

MAX_SEQ_LENGTH=128

TRAIN_BATCH_SIZE=4

PREDICT_BATCH_SIZE=4

LEARNING_RATE=5e-5

SAVE_CHECKPOINTS_STEPS=100

MAX_STEPS_WITHOUT_INCREASE=200

NUM_TRAIN_EPOCHS=2.3

# 输出路径
WORK_DIR=model_add_lstm

# 训练集与验证集大小, train_size + eval_size = 1
TRAIN_SIZE=1

# 是否进行模型融合
DO_ENSEMBLE=true

# 融合的模型个数
ENSEMBLE_NUM=4
