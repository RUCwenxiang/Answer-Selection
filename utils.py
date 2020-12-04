import pandas as pd
import numpy as np

def result_to_submission_file(output_predict_file=None, np_results=None):
    assert (not output_predict_file) ^ (not np_results), "output_predict_file or np_results only need one"
    # 需要用到样例提交的文件来进行修改结果文件
    source = pd.read_csv("sample_submission.tsv", header=None, sep="\t", names=["q_id", "a_id", "label"])
    if not output_predict_file:
        target = pd.read_csv(output_predict_file, header=None, sep="\t", names=["q_id", "a_id", "label"])
        source["label"] = target["label"]
    else:
        source["label"] = np_results
    source.to_csv("submission.tsv", header=None, sep='\t', index=False)

def ensemble_results_to_submission_file(model_name, model_num):
    final_arr = None
    for index in range(model_num):
        cur_arr = np.load("{}-{}/test_results.npy".format(model_name, model_num))
        if not final_arr:
            final_arr = cur_arr
        else:
            final_arr = np.vstack((final_arr, cur_arr))
    final_arr = (np.average(final_arr, axis=-1) > 0.5).astype(int)
    result_to_submission_file(np_results=final_arr)