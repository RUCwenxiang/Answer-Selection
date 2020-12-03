import sys
import pandas as pd
def result_file_to_submission_file(output_predict_file):
    # 需要用到样例提交的文件来进行修改结果文件
    source = pd.read_csv("sample_submission.tsv", header=None, sep="\t", names=["q_id", "a_id", "label"])
    target = pd.read_csv(output_predict_file, header=None, sep="\t", names=["q_id", "a_id", "label"])
    source["label"] = target["label"]
    source.to_csv("submission.tsv", header=None, sep='\t', index=False)

if __name__ == "__main__":
    file_path = sys.argv[1]
    result_file_to_submission_file(file_path)