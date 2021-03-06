import os
import random
import threading
import queue
# DATA_PATH + 'set_num_l.txt'
# DATA_PATH + 'set_num_r.txt'

class DataMgr(object):
    def __init__(self, data_path, batch_size, use_bern, embedding_size):
        self.use_bern = use_bern
        self.batch_size = batch_size
        self.embedding_size = embedding_size

        entity2id_data = os.path.join(data_path, 'entity2id.txt')
        relation2id_data = os.path.join(data_path, 'relation2id.txt')
        train_data = os.path.join(data_path, 'train.txt')
        entity2vec_data = os.path.join(data_path, 'entity2vec.bern')
        relation2vec_data = os.path.join(data_path, 'relation2vec.bern')
        sparse_index_head_data = os.path.join(data_path, 'set_num_l.txt')# l for left
        sparse_index_tail_data = os.path.join(data_path, 'set_num_r.txt') # r for right

        self._load_entity_and_relation_set(entity2id_data, relation2id_data)
        self._load_trilet_train_data(train_data)
        if self.use_bern:
            self._compute_prob_of_replace_head()
        self._load_entity_and_relation_embeddings(entity2vec_data, relation2vec_data)
        self._load_sparse_index(sparse_index_head_data, sparse_index_tail_data)

        # multithread feed
        self.lock = threading.Lock()
        self.batch_buf = queue.Queue()
        self.buf_sz = 72
        # self.buf_idx = buf_sz
        for i in range(self.buf_sz):
            batch = self.get_batch()
            self.batch_buf.put(batch)
            if i == self.buf_sz-1:
                print('batch_buf init finish.')
            # buf_idx = i



    def _load_entity_and_relation_set(self, entity2id_data, relation2id_data):
        # load entity set
        entity2id = {}
        id2entity = []
        with open(entity2id_data) as f:
            for line in f:
                entity, id = line.split()
                entity2id[entity] = int(id)
                id2entity.append(entity)
        # load relation set
        relation2id = {}
        id2relation = []
        with open(relation2id_data) as f:
            for line in f:
                relation, id = line.split()
                relation2id[relation] = int(id)
                id2relation.append(relation)

        self._entity2id = entity2id
        self._relation2id = relation2id
        self._id2entity = id2entity
        self._id2relation = id2relation

        self.entity_id_set = [i for i in range(len(id2entity))]
        self.relation_id_set = [i for i in range(len(id2relation))]

        self.relation_num = len(self.relation_id_set)
        self.entity_num = len(self.entity_id_set)

    def get_entity_id(self, entity):
        return self._entity2id[entity]

    def get_relation_id(self, relation):
        return self._relation2id[relation]

    def _load_trilet_train_data(self, train_file):
        # load triplet data
        tail = {} # given a relation and head to get tail
        head = {} # given a relation and tail to get head
        # head_counter = {}
        # tail_counter = {}
        train_data = []
        with open(train_file) as f:
            for line in f:
                h, r, t = line.split()

                hid = self.get_entity_id(h)
                rid = self.get_relation_id(r)
                tid = self.get_entity_id(t)

                train_data.append((hid, rid, tid))

                # if hid in head_counter:
                #     head_counter[hid] += 1
                # else:
                #     head_counter[hid] = 0
                # if tid in tail_counter:
                #     tail_counter[tid] += 1
                # else:
                #     tail_counter[tid] = 0

                if rid not in head:
                    head[rid] = {}
                    tail[rid] = {}
                if tid not in head[rid]:
                    head[rid][tid] = [hid]
                else:
                    head[rid][tid].append(hid)
                if hid not in tail[rid]:
                    tail[rid][hid] = [tid]
                else:
                    tail[rid][hid].append(tid)
        self.steps_per_epoch = len(train_data) // self.batch_size
        self._head = head
        self._tail = tail
        self.train_data = train_data

    def _compute_prob_of_replace_head(self):
        # Compute probability (Bernoulli distributaion) of replacing head used in
        # constructing negative tuplet. See the following paper for details.
        # http://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/viewFile/8531/8546
        prob_head = [0.5 for _ in self.relation_id_set]
        self._prob_head = prob_head
        if not self.use_bern:
            return

        for rid in self.relation_id_set:
            num_tails = 0
            num_heads = 0
            # FIXME
            for hid in self._tail[rid].keys():
                num_tails += len(self._tail[rid][hid])
                num_heads += 1
            tph = num_tails / num_heads # average tail number per head for a relation
            hpt = num_heads / num_tails # average head number per tail for a relation
            prob_head[rid] = tph / (hpt + tph)

    def _load_entity_and_relation_embeddings(self, entity2vec_data, relation2vec_data):
        # load embeddings
        entity_embeddings = []
        with open(entity2vec_data) as f:
            for line in f:
                embedding = [float(i) for i in line.split()]
                entity_embeddings.append(embedding)
        relation_embeddings = []
        with open(relation2vec_data) as f:
            for line in f:
                embedding = [float(i) for i in line.split()]
                relation_embeddings.append(embedding)

        self.entity_embeddings = entity_embeddings
        self.relation_embeddings = relation_embeddings

    def _load_sparse_index(self, sparse_index_head_data, sparse_index_tail_data):
        n = self.embedding_size
        r = self.relation_num

        sparse_index_head = []
        with open(sparse_index_head_data) as f:
            for line_num, line in enumerate(f):
                indices = line.split()
                rid = line_num // n
                row = line_num % n
                # nozero_num = int(indices[0])
                for col in indices[1:]:
                    sparse_index_head.append([rid, row, int(col)])

        sparse_index_tail = []
        with open(sparse_index_tail_data) as f:
            for line_num, line in enumerate(f):
                indices = line.split()
                rid = line_num // n
                row = line_num % n
                # nozero_num = int(indices[0])
                for col in indices[1:]:
                    sparse_index_tail.append([rid, row, int(col)])
        self.sparse_index_head = sparse_index_head
        self.sparse_index_tail = sparse_index_tail

    def get_batch(self):
        rids=[]
        hids=[]
        tids=[]
        n_hids=[]
        n_tids=[]
        flag_heads = []
        for _ in range(self.batch_size):
            pos_triplet = random.choice(self.train_data)
            h, r, t = pos_triplet
            nh, nt, replace_head = self._generate_negative_triplet(pos_triplet)

            rids.append(r)
            hids.append(h)
            tids.append(t)
            n_hids.append(nh)
            n_tids.append(nt)
            flag_heads.append(replace_head)
        return rids, hids, tids, n_hids, n_tids, flag_heads
    
    def get_batch_multi_thread(self):
        with self.lock:
            batch = self.batch_buf.get()
            if self.batch_buf.empty():
                for i in range(self.buf_sz):
                    batch_ = self.get_batch()
                    self.batch_buf.put(batch_)
            return batch

    def _generate_negative_triplet(self, pos_triplet):
        hid, rid, tid =  pos_triplet
        n_hid, _, n_tid = pos_triplet
        replace_head = None
        # probability of replace head
        prob = 0.5 # uniform distribution
        if self.use_bern:
            # Bernoulli distribution
            prob = self._prob_head[rid]

        if random.random() < prob:
            # replace head
            replace_head = True
            while n_hid in self._head[rid][tid]:
                n_hid = random.choice(self.entity_id_set)
        else:
            # replace tail
            replace_head = False
            while n_tid in self._tail[rid][hid]:
                n_tid = random.choice(self.entity_id_set)
        return n_hid, n_tid, replace_head


if __name__ == "__main__":
    data_mgr = DataMgr('data/', 100, True, 20)
    batch_data = data_mgr.get_batch()
    print(len(batch_data))
