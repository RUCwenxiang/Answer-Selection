import pandas as pd
import numpy as np

def result_to_submission_file(results_type, results):
    # 需要用到样例提交的文件来进行修改结果文件
    source = pd.read_csv("sample_submission.tsv", header=None, sep="\t", names=["q_id", "a_id", "label"])
    if results_type == "file":
        target = pd.read_csv(results, header=None, sep="\t", names=["q_id", "a_id", "label"])
        source["label"] = target["label"]
    else:
        source["label"] = results
    source.to_csv("submission.tsv", header=None, sep='\t', index=False)

def ensemble_results_to_submission_file(model_name, model_num):
    final_arr = None
    flag = False
    for index in range(model_num):
        cur_arr = np.load("{}-{}/test_results.npy".format(model_name, index))
        if not flag:
            final_arr = cur_arr
            flag = True
        else:
            final_arr = np.vstack((final_arr, cur_arr))
    final_arr = (np.average(final_arr, axis=0) > 0.5).astype(int)
    result_to_submission_file(results_type="numpy_array", results=final_arr)
