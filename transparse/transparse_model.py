import tensorflow as tf
import random

class TranSparseModel(object):
    def __init__(self, relation_num, entity_num, embedding_size,
                sparse_index_head, sparse_index_tail, lr,l1_norm = False):
        # Mh is Sparse Transfer Matrix of head for all relations.
        # Mh: r x m x n, m == n, r is relation num, n is embedding_size.
        # Mh[r] is Transfer Matrix for head for r-th relation.
        # Mh[r] is initialized as Identity Matrix.
        Mh = tf.Variable([tf.eye(embedding_size, dtype=tf.float32) for _ in range(relation_num)])
        # Mt is Sparse Transfer Matrix of tail for all relations
        # The shape of Mt is the same as Mh
        Mt = tf.Variable([tf.eye(embedding_size, dtype=tf.float32) for _ in range(relation_num)])

        # triplet input
        pos_triplet_id = tf.placeholder(tf.int32, shape=[3])
        neg_triplet_id = tf.placeholder(tf.int32, shape=[3])

        # embedding
        relation_embeddings = tf.placeholder(tf.float32, shape=[relation_num, embedding_size])
        entity_embeddings = tf.placeholder(tf.float32, shape=[entity_num, embedding_size])

        pos_triplet_embed = tf.nn.embedding_lookup(
            [entity_embeddings, relation_embeddings, entity_embeddings], pos_triplet_id
        )
        neg_triplet_embed = tf.nn.embedding_lookup(
            [entity_embeddings, relation_embeddings, entity_embeddings], neg_triplet_id
        )

        def score(head, relation, tail, rid):
            transfer_h = Mh[rid]
            transfer_t = Mt[rid]
            index_h = sparse_index_head[rid]
            index_t = sparse_index_tail[rid]

            # projected head: Mh * h
            p_head = tf.Variable(tf.zeros([embedding_size], dtype=tf.float32))
            rows = embedding_size
            for row in range(rows):
                for column in index_h[row]:
                    p_head[row] += transfer_h[row][column] * head[row]
            # projected tail: Mt * t
            p_tail = tf.Variable(tf.zeros([embedding_size], dtype=tf.float32))
            for row in range(rows):
                for column in index_t[row]:
                    p_tail[row] += transfer_t[row][column] * tail[row]
            if l1_norm:
                return tf.reduce_sum(tf.abs(p_head + relation - p_tail))
            else:
                return tf.reduce_sum(tf.square(p_head + relation - p_tail))

        # FIXME
        loss = score(pos_triplet_embed) - score(neg_triplet_embed) + margin


        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(self.loss)

    def step(self, session, x_train, y_train):
        summery, _, loss_val = session.run([self.merged, self.train, self.loss],
                {self.x_train: x_train, self.y_train: y_train})
        return summery, loss_val
