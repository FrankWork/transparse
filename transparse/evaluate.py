import numpy as np
import theano
import theano.tensor as T
import os
from dev_test import output_result

entity2id_fid = open('data/entity2id.txt')
entity2id = {}
count = 0
for e in entity2id_fid :
    con = e.strip('\n').split(' ')
    entity2id[con[0]] = count
    count += 1
entity2id_fid.close()

relation2id_fid = open('data/relation2id.txt')
relation2id = {}
count = 0
for e in relation2id_fid :
    con = e.strip('\n').split(' ')
    relation2id[con[0]] = count
    count += 1
relation2id_fid.close()



dev_fid = open('data/dev.txt')
dev_data = dev_fid.readlines()
dev_fid.close()
dev_triples = []
n_dev = len(dev_data)
dev_triples_idx = np.zeros((n_dev, 3),'int32')
relation_list = set()
count = 0
for dev_t in dev_data:
    con = dev_t.strip('\n')
    dev_triples.append(con)
    h, r, t, label = con.split('\t')
    h_id = entity2id[h]
    r_id = relation2id[r]
    t_id = entity2id[t]
    dev_triples_idx[count] = np.asarray([h_id, r_id, t_id],'int32')
    count += 1
    relation_list.add(r)


test_fid = open('data/test.txt')
test_data = test_fid.readlines()
test_fid.close()
test_triples = []
n_test = len(test_data)
test_triples_idx = np.zeros((n_test, 3),'int32')
count = 0
for test_t in test_data:
    con = test_t.strip('\n')
    test_triples.append(con)
    h, r, t, label = con.split('\t')
    h_id = entity2id[h]
    r_id = relation2id[r]
    t_id = entity2id[t]
    test_triples_idx[count] = np.asarray([h_id, r_id, t_id],'int32')
    count += 1
    relation_list.add(r)
relation_list = list(relation_list)
n_relation = len(relation_list)

def norm_matrix(matrix):
    return T.sum(T.sqr(matrix),axis=1)

def score():
    idx = T.imatrix('idx')
    ent_embed = T.matrix('ent_embed')
    rel_embed = T.matrix('rel_embed')
    A_h_mat = T.tensor3('A_h_mat')
    A_t_mat = T.tensor3('A_t_mat')
    h = ent_embed[idx[:,0]]
    r = rel_embed[idx[:,1]]
    t = ent_embed[idx[:,2]]
    A_h = A_h_mat[idx[:,1]]
    A_t = A_t_mat[idx[:,1]]
    res = T.batched_dot(A_t,t) -T.batched_dot(A_h, h) - r
    #score = -norm_matrix(res)
    score =-T.sum( T.abs_(res),axis=1)
    #res = h+r-t
    #score =-T.sum( T.abs_(res),axis=1)
    return theano.function(inputs = [idx, ent_embed, rel_embed , A_h_mat, A_t_mat],
                           outputs = score,on_unused_input='ignore')
     #theano.function(inputs = [idx, ent_embed, rel_embed],\
                           #outputs = score,on_unused_input='ignore')

score_f = score()

def eval(path, i):
    print('epoch %d:' % i)
    entity2vec = np.loadtxt(os.path.join(path, 'entity2vec.bern'+str(i)))
    relation2vec = np.loadtxt(os.path.join(path, 'relation2vec.bern'+str(i)))
    A_h = np.loadtxt(os.path.join(path, 'A_h.bern'+str(i))).reshape((n_relation,20,20))
    A_t = np.loadtxt(os.path.join(path, 'A_t.bern'+str(i))).reshape((n_relation,20,20))

    dev_score = score_f(dev_triples_idx, entity2vec, relation2vec, A_h, A_t)
    test_score = score_f(test_triples_idx, entity2vec, relation2vec, A_h, A_t)

    output_result(dev_triples, dev_score, test_triples, test_score, relation_list)

for epoch in range(3):
    eval('output/', epoch)
