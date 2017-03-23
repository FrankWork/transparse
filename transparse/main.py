import numpy as np
import theano
import theano.tensor as T
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

path = 'output/'
for i in [0]: #range(0,1000):
    print(i)
    entity2vec = np.loadtxt(path + 'entity2vec.bern'+str(i))
    relation2vec = np.loadtxt(path + 'relation2vec.bern'+str(i))
    A_h = np.loadtxt(path + 'A_h.bern'+str(i)).reshape((n_relation,20,20))
    A_t = np.loadtxt(path + 'A_t.bern'+str(i)).reshape((n_relation,20,20))

    dev_score = score_f(dev_triples_idx, entity2vec, relation2vec, A_h, A_t)
    test_score = score_f(test_triples_idx, entity2vec, relation2vec, A_h, A_t)

    output_result(dev_triples, dev_score, test_triples, test_score, relation_list)

# path = 'output-3-5/'
# 0
# _synset_domain_topic 622 -4.70953926165 0.9212218649517685
# _similar_to 42 -1.9920686234 0.5
# _type_of 5504 -4.81907122418 0.8566497093023255
# _part_of 1266 -4.06411194552 0.8933649289099526
# _domain_topic 116 -4.14558271404 0.7672413793103449
# _member_meronym 2268 -3.74206254609 0.9140211640211641
# _member_holonym 2346 -3.84976559671 0.9164535379369139
# _has_instance 6334 -4.29285197235 0.8214398484370067
# _has_part 1348 -4.07170310258 0.8857566765578635
# _domain_region 592 -3.43528640385 0.7668918918918919
# _subordinate_instance_of 650 -3.11798919261 0.8892307692307693
# 0.8621490895295902
# 999
# _synset_domain_topic 622 -4.87059401444 0.9260450160771704
# _similar_to 42 -1.63501570595 0.5
# _type_of 5504 -4.97759029287 0.8551962209302325
# _part_of 1266 -4.16587453819 0.8878357030015798
# _domain_topic 116 -2.47804494414 0.7672413793103449
# _member_meronym 2268 -3.94376759031 0.9157848324514991
# _member_holonym 2346 -3.72219630468 0.9156010230179028
# _has_instance 6334 -4.50463573285 0.8179665298389643
# _has_part 1348 -4.05371212959 0.884272997032641
# _domain_region 592 -3.74063951709 0.793918918918919
# _subordinate_instance_of 650 -3.58832802098 0.8923076923076924
# 0.8613903641881638


# path = '1000-epoch-norm-p-output-3-20/'
# 0
# _member_holonym 2346 -20.4554983149 0.5741687979539642
# _similar_to 42 -15.4022061701 0.47619047619047616
# _type_of 5504 -19.4532545371 0.520530523255814
# _domain_region 592 -11.3660863646 0.5
# _has_part 1348 -11.5880764396 0.49851632047477745
# _synset_domain_topic 622 -21.8771393768 0.4887459807073955
# _member_meronym 2268 -25.0166209631 0.5052910052910053
# _domain_topic 116 -14.3937465124 0.5086206896551724
# _part_of 1266 -18.740088403 0.5750394944707741
# _has_instance 6334 -15.6599802968 0.523523839595832
# _subordinate_instance_of 650 -19.3162518825 0.5184615384615384
# 0.525891502276176
# 999
# _member_holonym 2346 -3.89898291505 0.4982949701619778
# _similar_to 42 -63.434194059 0.47619047619047616
# _type_of 5504 -43.2852903693 0.7013081395348837
# _domain_region 592 -6.36524197126 0.5118243243243243
# _has_part 1348 -44.0087386527 0.5979228486646885
# _synset_domain_topic 622 -81.0658725627 0.48392282958199356
# _member_meronym 2268 -54.4341967115 0.5925925925925926
# _domain_topic 116 -12.1535229029 0.4396551724137931
# _part_of 1266 -6.92104711564 0.5126382306477093
# _has_instance 6334 -22.4267724446 0.5361540890432586
# _subordinate_instance_of 650 -46.9075860876 0.6738461538461539
# 0.5850246585735963
