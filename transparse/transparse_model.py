import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np

class TranSparseModel(object):
    def __init__(self, margin, lr, l1_norm,
                relation_num, entity_num,
                embedding_size, batch_size,
                entity_embed, relation_embed,
                sparse_index_head, sparse_index_tail):
        with tf.name_scope("TransparseModel"):
            with tf.name_scope("Transfer_Matrix"):
                # Mh is Sparse Transfer Matrix of head for all relations.
                # Mh: r x m x n, m == n, r is relation num, n is embedding_size.
                # Mh[r] is Transfer Matrix for head for r-th relation.
                # Mh[r] is initialized as Identity Matrix.
                # Mt is Sparse Transfer Matrix of tail for all relations
                # The shape of Mt is the same as Mh
                M_init = np.asarray(
                    [np.eye(embedding_size) for _ in range(relation_num)]
                )
                Mh = tf.Variable(M_init, dtype=tf.float32, name="Mh")
                Mt = tf.Variable(M_init, dtype=tf.float32, name="Mt")

                shape = [relation_num, embedding_size, embedding_size]
                mask_h = tf.sparse_to_dense(sparse_index_head, shape,
                                            tf.ones([len(sparse_index_head)]),
                                            name='mask_h')
                mask_t = tf.sparse_to_dense(sparse_index_tail, shape,
                                            tf.ones([len(sparse_index_tail)]),
                                            name='mask_t')

            with tf.name_scope('Input'):
                # mini-batch input, shape: b, b is batch_size
                rids = tf.placeholder(tf.int32,
                                        shape=[batch_size],
                                        name='rids')
                hids = tf.placeholder(tf.int32,
                                        shape=[batch_size],
                                        name='hids')
                tids = tf.placeholder(tf.int32,
                                        shape=[batch_size],
                                        name='tids')
                n_hids = tf.placeholder(tf.int32,
                                        shape=[batch_size],
                                        name='n_hids')
                n_tids = tf.placeholder(tf.int32,
                                        shape=[batch_size],
                                        name='n_tids')
                flag_heads = tf.placeholder(tf.bool,
                                            shape=[batch_size],
                                            name='flag_heads')

            with tf.name_scope('Embedding'):
                # relation and entity embedding
                # relation_embeddings shape: r x m
                # entity_embeddings: shape: r x n
                if entity_embed is None and relation_embed is None:
                    relation_embeddings = tf.Variable(
                                    tf.truncated_normal([relation_num,
                                                embedding_size, 1]),
                                    name='relation_embeddings')
                    entity_embeddings = tf.Variable(
                                    tf.truncated_normal([entity_num,
                                                embedding_size, 1]),
                                    name='entity_embeddings')
                else:
                    relation_embed = tf.reshape(relation_embed,
                                    [relation_num, embedding_size, 1])
                    relation_embeddings = tf.Variable(relation_embed,
                                            dtype=tf.float32,
                                            name='relation_embeddings')
                    entity_embed = tf.reshape(entity_embed,
                                    [entity_num, embedding_size, 1])
                    entity_embeddings = tf.Variable(entity_embed,
                                            dtype=tf.float32,
                                            name='entity_embeddings')

            with tf.name_scope('Embedding_Lookup'):
                r_embed = tf.nn.embedding_lookup(relation_embeddings,
                                            rids, name='r_embed')
                h_embed = tf.nn.embedding_lookup(entity_embeddings,
                                            hids, name='h_embed')
                t_embed = tf.nn.embedding_lookup(entity_embeddings,
                                            tids, name='t_embed')
                neg_h_embed = tf.nn.embedding_lookup(entity_embeddings,
                                            n_hids,name='neg_h_embed')
                neg_t_embed = tf.nn.embedding_lookup(entity_embeddings,
                                            n_tids,name='neg_t_embed')
                Mh_batch = tf.nn.embedding_lookup(Mh, rids,
                                            name='Mh_batch')
                Mt_batch = tf.nn.embedding_lookup(Mt, rids,
                                            name='Mt_batch')
                mask_h_batch = tf.nn.embedding_lookup(mask_h, rids,
                                            name='mask_h_batch')
                mask_t_batch = tf.nn.embedding_lookup(mask_t, rids,
                                            name='mask_t_batch')

            with tf.name_scope('Loss'):
                # projected heads and tails

                # h_p = tf.matmul(Mh_batch, h_embed, a_is_sparse=True, name='h_p')
                # t_p = tf.matmul(Mt_batch, t_embed, a_is_sparse=True, name='t_p')
                # neg_h_p = tf.matmul(Mh_batch, neg_h_embed, a_is_sparse=True, name='neg_h_p')
                # neg_t_p = tf.matmul(Mt_batch, neg_t_embed, a_is_sparse=True, name='neg_t_p')

                h_p = tf.matmul(Mh_batch, h_embed, name='h_p')
                t_p = tf.matmul(Mt_batch, t_embed, name='t_p')
                neg_h_p = tf.matmul(Mh_batch, neg_h_embed,  name='neg_h_p')
                neg_t_p = tf.matmul(Mt_batch, neg_t_embed,  name='neg_t_p')
                score_pos = h_p + r_embed - t_p
                score_neg = neg_h_p + r_embed - neg_t_p
                if l1_norm:
                    score_pos = tf.abs(score_pos)
                    score_neg = tf.abs(score_neg)
                else:
                    score_pos = tf.square(score_pos)
                    score_neg = tf.square(score_neg)

                logits = tf.maximum(margin + score_pos - score_neg, 0, name='logits')
                loss = tf.reduce_sum(logits, name='loss')

                tf.summary.scalar('loss', loss)

            with tf.name_scope('Optimizer'):
                optimizer = tf.train.GradientDescentOptimizer(lr)
                grad_val = optimizer.compute_gradients(loss, [Mh, Mt,
                                        relation_embeddings, entity_embeddings])

                gh, vh = grad_val[0]
                gt, vt = grad_val[1]

                # gh is IndexedSlices
                # mask_h_batch is Tensor
                mask_gh = tf.IndexedSlices(
                                    tf.multiply(gh.values, mask_h_batch),
                                    gh.indices, gh.dense_shape
                                    )
                mask_gt = tf.IndexedSlices(
                                    tf.multiply(gt.values, mask_t_batch),
                                    gt.indices, gt.dense_shape
                                    )

                mask_grad_val = [(mask_gh, vh),
                                (mask_gt, vt),
                                grad_val[2],
                                grad_val[3]
                                ]
                # grad_val = [(tf.clip_by_value(grad, -0.5, 0.5), var) for grad, var in grad_val]

                # mask_grad_val = grad_val
                train_op = optimizer.apply_gradients(mask_grad_val)

            with tf.name_scope('Norm'):
                def norm_op(embeddings, ids, embed):
                    norm = tf.norm(embed, axis=1, keep_dims=True)
                    indices = tf.where(tf.greater(norm, 1))

                    gather_embed = tf.gather(embed, indices)
                    gather_norm = tf.gather(norm, indices)
                    gather_ids = tf.gather(ids, indices)

                    normed_embed = gather_embed / gather_norm
                    return tf.scatter_update(embeddings,
                                             gather_ids,
                                             normed_embed)

                h_norm_op = norm_op(entity_embeddings, hids, h_embed)
                t_norm_op = norm_op(entity_embeddings, tids, t_embed)
                n_ids = tf.where(flag_heads, n_hids, n_tids)
                n_embed = tf.where(flag_heads, neg_h_embed, neg_t_embed)
                neg_norm_op = norm_op(entity_embeddings, n_ids, n_embed)
                with tf.control_dependencies([h_norm_op,
                                              t_norm_op,
                                              neg_norm_op]):
                    r_norm_op = norm_op(relation_embeddings,
                                        rids,
                                        r_embed)

            with tf.name_scope('Projected'):
                def body(_):
                    with tf.name_scope('Embedding_Lookup'):
                        h_embed = tf.nn.embedding_lookup(entity_embeddings,
                                                    hids, name='h_embed')
                        t_embed = tf.nn.embedding_lookup(entity_embeddings,
                                                    tids, name='t_embed')
                        neg_h_embed = tf.nn.embedding_lookup(entity_embeddings,
                                                    n_hids,name='neg_h_embed')
                        neg_t_embed = tf.nn.embedding_lookup(entity_embeddings,
                                                    n_tids,name='neg_t_embed')
                        Mh_batch = tf.nn.embedding_lookup(Mh, rids,
                                                    name='Mh_batch')
                        Mt_batch = tf.nn.embedding_lookup(Mt, rids,
                                                    name='Mt_batch')
                    with tf.name_scope('Projected'):
                        h_p = tf.matmul(Mh_batch, h_embed, name='h_p')
                        t_p = tf.matmul(Mt_batch, t_embed, name='t_p')
                        neg_h_p = tf.matmul(Mh_batch, neg_h_embed,  name='neg_h_p')
                        neg_t_p = tf.matmul(Mt_batch, neg_t_embed,  name='neg_t_p')
                        neg_p = tf.where(flag_heads, neg_h_p, neg_t_p)
                    with tf.name_scope('Loss'):
                        loss_p = tf.reduce_sum(
                            tf.maximum(tf.reduce_sum(tf.square(h_p))-1, 0) + \
                            tf.maximum(tf.reduce_sum(tf.square(t_p))-1, 0) + \
                            tf.maximum(tf.reduce_sum(tf.square(neg_p))-1, 0)
                            )
                    with tf.name_scope('Gradients'):
                        grad_val_p = optimizer.compute_gradients(loss_p)
                        # NOTE: if there is a `NoneType` grad in grad_val pair,
                        # the entire grad_val list is not fetchable.
                        grad_val_p = [(g,v) for g,v in grad_val_p if g is not None]
                        gh, vh = grad_val_p[0]
                        gt, vt = grad_val_p[1]
                        mask_gh = tf.IndexedSlices(
                                            tf.multiply(gh.values, mask_h_batch),
                                            gh.indices, gh.dense_shape
                                            )
                        mask_gt = tf.IndexedSlices(
                                            tf.multiply(gt.values, mask_t_batch),
                                            gt.indices, gt.dense_shape
                                            )
                        mask_grad_val_p = [(mask_gh, vh),
                                        (mask_gt, vt),
                                        grad_val[2]
                                        ]
                        train_p_op = optimizer.apply_gradients(mask_grad_val_p)

                    # tf_print = tf.Print(loss_p, [loss_p])
                    with tf.control_dependencies([train_p_op]):#tf_print,
                        return tf.cond(tf.greater(loss_p, 0.) ,
                                        lambda : tf.assign(flag, True),
                                        lambda : tf.assign(flag, False)
                                        )
                flag = tf.Variable(True, trainable=False, dtype=tf.bool)
                # Value of flag is True in Epoch 0, and False in other Epoch
                with tf.control_dependencies([tf.assign(flag, True)]):
                    while_op = tf.while_loop(lambda x: x, body, [flag])

        self.merged = tf.summary.merge_all()
        self.global_step = tf.Variable(initial_value=0, trainable=False)
        self.saver = tf.train.Saver(tf.global_variables())

        self.rids = rids
        self.hids = hids
        self.tids = tids
        self.n_hids = n_hids
        self.n_tids = n_tids
        self.flag_heads = flag_heads

        self.Mh = Mh
        self.Mt = Mt
        self.relations = relation_embeddings
        self.entitys = entity_embeddings

        self.logits = logits
        self.loss = loss
        self.train_op = train_op
        self.norm_op = r_norm_op
        self.gvs = mask_grad_val

        self.train_p_op = while_op#train_p_op

    def train_minibatch(self, session, rids, hids, tids, n_hids,
                                                    n_tids, flag_heads):
        feed_dict = {self.rids:rids, self.hids:hids, self.tids:tids,
                     self.n_hids:n_hids, self.n_tids:n_tids,
                     self.flag_heads:flag_heads}
        summary, _, _, _, loss_val = session.run([self.merged,
                     self.train_op, self.norm_op, self.train_p_op, self.loss],
                     feed_dict=feed_dict)
        return summary, loss_val

    def get_matrix_and_embeddings(self, session):
        return session.run([self.Mh, self.Mt, self.relations, self.entitys])

    def test_optimizer(self, session, rids, hids, tids, n_hids, n_tids, flag_heads):
        feed_dict={self.rids:rids, self.hids:hids, self.tids:tids,
            self.n_hids:n_hids, self.n_tids:n_tids,
            self.flag_heads:flag_heads
        }
        _, _, _, loss, gvs = session.run([self.train_op, self.norm_op,
                            self.train_p_op, self.loss, self.gvs], feed_dict)
        return loss, gvs
