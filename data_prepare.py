# -*- coding: UTF-8 -*-
from collections import defaultdict
from random import random

def produce_train_data(train_query_path, train_relay_path, train_path, eval_path):

    querys = {}
    with open(train_query_path, encoding="utf-8") as f:
        for line in f:
            query_id, query = line.strip().split('\t')
            querys[query_id] = query

    query_replays = defaultdict(dict)
    max_answer_num = 0
    sequence_lengths = []
    with open(train_relay_path, encoding="utf-8") as f:
        for line in f:
            query_id, replay_id, replay, label = line.strip().split('\t')
            max_answer_num = max(max_answer_num, int(replay_id)+1) # 记录最大答案个数
            sequence_lengths.append(len(query) + len(replay) + 3) # 记录序列长度, 加上了[CLS], [SEP](2个)
            query = querys[query_id] # 原查询
            if replay_id == '0':
                query_replays[query_id]['Sentences'] = []
                query_replays[query_id]['Labels'] = []
            query_replays[query_id]['Sentences'].append(query + '&&&&&' + replay)
            query_replays[query_id]['Labels'].append(label)

    print("max_answer_num: {}".format(max_answer_num))
    print("sequence_lengths: {}".format(sorted(sequence_lengths)))
#    for query_id in range(3):
#        print("query_id: ", query_id, query_replays[str(query_id)])

    with open(train_path, 'w', encoding="utf-8") as f_train, open(eval_path, 'w', encoding="utf-8") as f_eval:
        for query_id in query_replays.keys():
            sentences = query_replays[query_id]['Sentences']
            labels = query_replays[query_id]['Labels']
            instance = "#####".join(sentences) + '|||||'.join(['', str(len(labels)), ' '.join(labels)]) + "\n"
            if random() < 0.9:
                f_train.write(instance)
            else:
                f_eval.write(instance)

if __name__ == "__main__":
    train_query_path = "data/train/train.query.tsv"
    train_relay_path = "data/train/train.reply.tsv"
    train_path = "data/train/train.txt"
    eval_path = "data/eval/eval.txt"
    produce_train_data(train_query_path, train_relay_path, train_path, eval_path)
