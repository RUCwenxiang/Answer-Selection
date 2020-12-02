nohup bash run.sh >run.log 2>&1 &
nohup bash run_mac_bert.sh >run_mac_bert.log 2>&1 &
grep -E "INFO:tensorflow:Saving dict|INFO:tensorflow:Loss" run.log
grep -E "INFO:tensorflow:Saving dict|INFO:tensorflow:Loss" run_mac_bert.log
grep -E "INFO:tensorflow:Saving dict|INFO:tensorflow:Loss" run.log >train_and_eval.result
