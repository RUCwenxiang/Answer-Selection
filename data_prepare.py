# -*- coding: UTF-8 -*-
import sys
from collections import defaultdict
from random import random


def produce_train_data(query_path, relay_path, train_path, eval_path, test_path, mode, train_size):
    querys = {}
    with open(query_path, encoding="utf-8") as f:
        for line in f:
            query_id, query = line.strip().split('\t')
            querys[query_id] = query

    query_replays = defaultdict(dict)
    max_answer_num = 0
    sequence_lengths = []
    with open(relay_path, encoding="utf-8") as f:
        for line in f:
            if mode == "train":
                query_id, replay_id, replay, label = line.strip().split('\t')
            else:
                query_id, replay_id, replay = line.strip().split('\t')
            max_answer_num = max(max_answer_num, int(replay_id)+1) # 记录最大答案个数
            query = querys[query_id] # 原查询
            sequence_lengths.append(len(query) + len(replay) + 3)  # 记录序列长度, 加上了[CLS], [SEP](2个)
            if replay_id == '0':
                query_replays[query_id]['Sentences'] = []
                if mode == "train":
                    query_replays[query_id]['Labels'] = []
            query_replays[query_id]['Sentences'].append(query + '&&&&&' + replay)
            if mode == "train":
                query_replays[query_id]['Labels'].append(label)

    print("max_answer_num: {}".format(max_answer_num))
    print("max longest 200 sequence_lengths: {}".format(sorted(sequence_lengths[:200])))
    for query_id in range(3):
       print("query_id: ", query_id, query_replays[str(query_id)])

    if mode == "train":
        with open(train_path, 'w', encoding="utf-8") as f_train, open(eval_path, 'w', encoding="utf-8") as f_eval:
            for query_id in query_replays.keys():
                sentences = query_replays[query_id]['Sentences']
                labels = query_replays[query_id]['Labels']
                instance = "#####".join(sentences) + '|||||'.join(['', str(len(labels)), ' '.join(labels)]) + "\n"
                if random() < train_size:
                    f_train.write(instance)
                else:
                    f_eval.write(instance)
    else:
        with open(test_path, 'w', encoding="utf-8") as f_test:
            for query_id in range(len(query_replays)):
                sentences = query_replays[str(query_id)]['Sentences']
                instance = "#####".join(sentences) + '|||||'.join(['', str(len(sentences))]) + "\n"
                f_test.write(instance)

if __name__ == "__main__":
    train_size = float(sys.argv[1])
    produce_train_data(query_path="data/query.tsv", relay_path="data/reply.tsv",
                      train_path="data/train/train.txt", eval_path="data/eval/eval.txt",
                      test_path=None, mode="train", train_size=train_size)
    produce_train_data(query_path="data/test/test.query.tsv", relay_path="data/test/test.reply.tsv", train_path=None,
                       eval_path=None, test_path="data/test/test.txt", mode="test", train_size=None)
