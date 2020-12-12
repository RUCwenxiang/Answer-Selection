> ### BERT based answer selection model
- 模型框架  
  - Roberta/Bert/Macbert + Bi-LSTM + CRF/Softmax
- 模型融合
  - 简单的模型ensemble, 就是生成多个模型取每个模型预测的概率值的平均值作为最后的预测, 最后一次提交用了四个模型, 如果生成8个模型之后再用LightGBM/XGBoost进行拟合最后的得分会更高
- 流程
  - 生成训练集, 测试集, 设置一个很大的epoch进行early stopping选一个合适的epoch值
  - 之后按照epoch的值训练多个模型
  - 对多个模型的预测进行简单的平均融合, 上传结果到obs
