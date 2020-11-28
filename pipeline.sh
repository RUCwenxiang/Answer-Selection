bash run.sh 2>&1 >run.log
grep -E "INFO:tensorflow:Saving dict|INFO:tensorflow:Loss" run.log >train_and_eval.result