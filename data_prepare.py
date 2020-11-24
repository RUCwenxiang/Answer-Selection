from collections import defaultdict

def produce_train_data(train_query_path, train_relay_path, train_path):

    querys = {}
    with open(train_query_path) as f:
        for line in f:
            query_id, query = line.strip().split('\t')
            querys[query_id] = query

    query_replays = defaultdict(dict)
    max_answer_num = 0
    with open(train_relay_path) as f:
        for line in f:
            query_id, replay_id, replay, label = line.strip().split('\t')
            max_answer_num = max(max_answer_num, int(replay_id)+1) # 记录最大答案个数
            query = querys[query_id] # 原查询
            if replay_id == '0':
                query_replays[query_id]['Sentences'] = []
                query_replays[query_id]['Labels'] = []
            query_replays[query_id]['Sentences'].append(query + '&&&&&' + replay)
            query_replays[query_id]['Labels'].append(label)

    print("最大答案个数: {}".format(max_answer_num))
    for query_id in range(3):
        print("query_id: ", query_id, query_replays[str(query_id)])

    with open(train_path, 'w') as f:
        for query_id in query_replays.keys():
            sentences = query_replays[query_id]['Sentences']
            labels = query_replays[query_id]['Labels']
            f.write("#####".join(sentences) + '|||||'.join(['', str(len(labels)), ' '.join(labels)]) + "\n")

if __name__ == "__main__":
    train_query_path = "data/train/train.query.tsv"
    train_relay_path = "data/train/train.reply.tsv"
    train_path = "data/train/train.txt"
    produce_train_data(train_query_path, train_relay_path, train_path)