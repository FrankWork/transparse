import numpy as np
# import cPickle

# dev
def thresh(score, label, interval_value = 0.01):
    score = np.asarray(score)
    max_score = np.max(score)
    min_score = np.min(score)
    n_interval = int((max_score - min_score) / interval_value)
    best_acc = 0
    best_thresh = 0
    predict_label = [None] * len(label)
    for i in range(n_interval):
        tmp_thresh = min_score + interval_value * i
        count = 0
        for j in range(len(label)):
            if(score[j] > tmp_thresh):
                predict_label[j] = '1'
            else:
                predict_label[j] = '-1'

        for j in range(len(label)):
            if(predict_label[j] == label[j]):
                count += 1
        acc = 1. * count / len(label)

        if(acc >= best_acc):
            best_acc = acc
            best_thresh = tmp_thresh

    return len(label), best_thresh, best_acc



def test(score, label, thresh):
    predict_label = [None] * len(label)
    count = 0
    for j in range(len(label)):
        if(score[j] > thresh):
            predict_label[j] = '1'
        else:
             predict_label[j] = '-1'
    for j in range(len(label)):
        if(predict_label[j] == label[j]):
            count += 1
    acc = 1. * count / len(label)

    return len(label), acc


def output_result(dev_triples, dev_score, test_triples, test_score, relation_list, interval=0.01):
    dev_relation_dic = {}
    test_relation_dic = {}
    for rel in relation_list:
        dev_relation_dic[rel]=[[],[]]
        test_relation_dic[rel]=[[],[]]

    count = 0
    for e in dev_triples:
        content = e.split('\t')
        relation = content[1]
        dev_relation_dic[relation][0].append(content[3])
        dev_relation_dic[relation][1].append(dev_score[count])
        count += 1

    count  = 0
    for e in test_triples:
        content = e.split('\t')
        relation = content[1]
        test_relation_dic[relation][0].append(content[3])
        test_relation_dic[relation][1].append(test_score[count])
        count += 1

        test_rel_acc = {}
        test_total_n = 0
        test_acc_total_n = 0

        test_number = len(test_triples)
        acc_test_number = 0

    for rel in relation_list:
         rel_dev_number, rel_dev_best_thresh, rel_dev_acc  = thresh(dev_relation_dic[rel][1], dev_relation_dic[rel][0])
         rel_test_number, rel_acc = test(test_relation_dic[rel][1], test_relation_dic[rel][0], rel_dev_best_thresh)
         acc_test_number += rel_test_number * rel_acc
         print(rel, rel_test_number, rel_dev_best_thresh, rel_acc)

    print(acc_test_number / test_number)
    return acc_test_number / test_number
