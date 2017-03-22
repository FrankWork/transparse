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
                                            tf.ones([len(sparse_index_head)]))
                mask_t = tf.sparse_to_dense(sparse_index_tail, shape,
                                            tf.ones([len(sparse_index_tail)]))

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
                h_p = tf.matmul(Mh_batch, h_embed)
                t_p = tf.matmul(Mt_batch, t_embed)
                neg_h_p = tf.matmul(Mh_batch, neg_h_embed)
                neg_t_p = tf.matmul(Mt_batch, neg_t_embed)
                score_pos = h_p + r_embed - t_p
                score_neg = neg_h_p + r_embed - neg_t_p
                if l1_norm:
                    score_pos = tf.abs(score_pos)
                    score_neg = tf.abs(score_neg)
                else:
                    score_pos = tf.square(score_pos)
                    score_neg = tf.square(score_neg)

                logits = tf.maximum(margin + score_pos - score_neg, 0)
                loss = tf.reduce_sum(logits)

                tf.summary.scalar('loss', loss)

            with tf.name_scope('Optimizer'):
                optimizer = tf.train.GradientDescentOptimizer(lr)
                grad_val = optimizer.compute_gradients(loss, [Mh, Mt,
                                        relation_embeddings, entity_embeddings])
                gh, vh = grad_val[0]
                gt, vt = grad_val[1]
                mask_grad_val = [(tf.multiply(gh, mask_h_batch), vh),
                                (tf.multiply(gt, mask_t_batch), vt),
                                grad_val[2], grad_val[3]]
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

            # with tf.name_scope('Norm_Projected'):
            #     neg_p = tf.where(flag_heads, neg_h_p, neg_t_p)
            #     loss_p = tf.reduce_sum(tf.maximum(h_p-1, 0) + \
            #                            tf.maximum(t_p-1, 0) + \
            #                            tf.maximum(neg_p-1, 0))
            #
            #     optimizer_p = tf.train.GradientDescentOptimizer(lr)
            #
            #     grad_val_p = optimizer_p.compute_gradients(loss_p,
            #                                         var_list=[Mh, Mt,
            #                                             relation_embeddings,
            #                                             entity_embeddings])
            #     # gh, vh = grad_val_p[0]
            #     # gt, vt = grad_val_p[1]
            #     # mask_grad_val_p = [(tf.multiply(gh, mask_h_batch), vh),
            #     #                 (tf.multiply(gt, mask_t_batch), vt),
            #     #                 grad_val_p[2], grad_val_p[3]]
            #     mask_grad_val_p = grad_val_p
            #     train_p_op = optimizer_p.apply_gradients(mask_grad_val_p)

        self.merged = tf.summary.merge_all()
        self.global_step = tf.Variable(initial_value=0, trainable=False)
        self.saver = tf.train.Saver(tf.global_variables())

        self.Mh = Mh
        self.Mt = Mt
        self.mask_h = mask_h
        self.mask_t = mask_t
        self.relations = relation_embeddings
        self.entitys = entity_embeddings

        self.rids = rids
        self.hids = hids
        self.tids = tids
        self.n_hids = n_hids
        self.n_tids = n_tids
        self.flag_heads = flag_heads

        self.logits = logits
        self.loss = loss
        self.train_op = train_op
        self.grad_val = mask_grad_val
        self.norm_op = r_norm_op

        # self.loss_p = loss_p
        # self.train_p_op = train_p_op
        # self.grad_val_p = mask_grad_val_p


    def train_minibatch(self, session, rids, hids, tids, n_hids,
                                                    n_tids, flag_heads):
        feed_dict = {self.rids:rids, self.hids:hids, self.tids:tids,
                     self.n_hids:n_hids, self.n_tids:n_tids,
                     self.flag_heads:flag_heads}
        summary, _, _, loss_val = session.run([self.merged,
                     self.train_op, self.norm_op, self.loss],
                     feed_dict=feed_dict)
        _, loss_p = session.run([self.train_p_op, self.loss_p],
                             feed_dict=feed_dict)
        while(loss_p > 0):
            _, loss_p = session.run([self.train_p_op, self.loss_p],
                                 feed_dict=feed_dict)
        return summary, loss_val

    def get_matrix_and_embeddings(self, session):
        Mh, Mt, relations, entitys = session.run([self.Mh, self.Mt,
                                                self.relations, self.entitys])
        return Mh, Mt, relations, entitys
    def get_mask(self, session):
        mask_h, mask_t = session.run([self.mask_h, self.mask_t])
        return mask_h, mask_t

    def test_minibatch(self, session, rids, hids, tids, n_hids, n_tids, flag_heads):
        feed_dict={self.rids:rids, self.hids:hids, self.tids:tids,
            self.n_hids:n_hids, self.n_tids:n_tids, self.flag_heads:flag_heads}
        # fetch_list = [self.train_op, self.norm_op, self.logits, self.loss, self.Mh, self.Mt,
        #                 self.relations, self.entitys, self.grad_val]
        # _, _, logits, loss, Mh, Mt, r, e, grad_val = session.run(fetch_list, feed_dict)

        _, loss_p1, grad_val_p1= session.run(#
                                    [self.train_p_op, self.loss_p, self.grad_val_p],#
                                    feed_dict
                                    )
        print(loss_p1)
        # for g,v in grad_val_p1:
        #     print('*' * 80)
        #     print(v)
        #     print('*' * 80)
        #     print(g)

        # while(loss_p > 0):
        #     _, loss_p, grad_val_p= session.run([self.train_p_op, self.loss_p, self.grad_val_p],
        #                          feed_dict=feed_dict)
        #     for g,v in grad_val_p:
        #         print('*' * 80)
        #         print(v)
        #         print('*' * 80)
        #         print(g)

        # return logits, loss ,Mh, Mt, r, e, grad_val


def test_update_norm():
    h_in = [0, 1]
    h_num = 3
    embedding_size = 3

    embeddings = tf.Variable(2  * tf.ones([h_num, embedding_size, 1]))
    h = tf.placeholder(tf.int32, [2])
    embed = tf.nn.embedding_lookup(embeddings, h)

    def norm_op(embeddings, ids, embed):
        norm = tf.norm(embed, axis=1, keep_dims=True)
        indices = tf.where(tf.greater(norm, 1))

        gather_embed = tf.gather(embed, indices)
        gather_norm = tf.gather(norm, indices)
        gather_ids = tf.gather(ids, indices)

        normed_embed = gather_embed / gather_norm
        return norm, normed_embed, tf.scatter_update(embeddings, gather_ids, normed_embed)
        # return norm, normed_embed, tf.cond(norm > 1, fn1, lambda: embed)

    norm, normed_embed, update_op = norm_op(embeddings, h, embed)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        _, e, n, ne, eds = session.run([update_op, embed, norm, normed_embed, embeddings], feed_dict={h:h_in})

        print(e)
        print('*' * 40)
        print(n)
        print('*' * 40)
        print(ne)
        print('*' * 40)
        print(eds)
        print('=' * 80)

def test_boolean_index():
    # tf.less
    # tf.gather
    # tf.where
    # tf.cond
    val = tf.random_uniform([10], minval=0, maxval=9, dtype=tf.float32)
    less = tf.less(val, 5)
    # indices = tf.range(10)
    indices = tf.where(less)
    val5 = tf.gather(val, indices)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        val = session.run([val, less, indices, val5])
        for v in val:
            print(v)
            print('*' * 40)

def test_concat_ids():
    flags = tf.Variable([True, False, True], dtype=tf.bool)
    ones = tf.Variable(tf.ones([3], dtype=tf.float32))
    zeros = tf.Variable(tf.zeros([3],dtype=tf.float32))
    val = tf.where(flags, ones, zeros)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        val = session.run([val])
        print('*' * 80)
        print(val)

def test_matmul():
    # @tf.RegisterGradient("CustomMatmul")
    # def _custom_matmul_grad(op, grad):
    #     # grad = tf.gradients(op.outputs, op.inputs)
    #     mask = tf.eye(3)
    #     print(tf.grad_fn(op, grad))
    #     return [op.inputs[0], op.inputs[1]]

    with tf.Graph().as_default() as g:
        s = tf.SparseTensor(indices=[[0,0],[1,1],[2,2]], values=[1., 1., 1.], dense_shape=[3,3])
        mask = tf.sparse_tensor_to_dense(s)
        w = tf.Variable(np.eye(3), dtype=tf.float32)
        x = tf.constant([1,2,3], dtype=tf.float32, shape=[3,1])
        # with g.gradient_override_map({"SparseMatMul": "CustomMatmul"}):
        logits = tf.matmul(w, x, a_is_sparse=True)
        loss = logits
        optimizer = tf.train.GradientDescentOptimizer(0.1)#.minimize(loss)
        grad_val = optimizer.compute_gradients(loss)
        # mask = tf.eye(3)
        grad_val = [(tf.multiply(grad_val[0][0], mask), grad_val[0][1])]
        train_op = optimizer.apply_gradients(grad_val)

    with tf.Session(graph=g) as session:
        # session = tf_debug.LocalCLIDebugWrapperSession(session)
        # session.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        session.run(tf.global_variables_initializer())
        _, g_v, w = session.run([train_op, grad_val, w])

        for g, v in g_v:
            print(g)
            print('*' * 80)
            print(v)
    # [[ 1.  0.  0.]
    #  [ 0.  2.  0.]
    #  [ 0.  0.  3.]]
    # ********************************************************************************
    # [[ 0.89999998  0.          0.        ]
    #  [ 0.          0.80000001  0.        ]
    #  [ 0.          0.          0.69999999]]

def test_square():
    @tf.RegisterGradient("CustomSquare")
    def _custom_square_grad(op, grad):
      return tf.constant([101.0])

    with tf.Graph().as_default() as g:
      c = tf.Variable([5.0], dtype=tf.float32)
      s_1 = tf.square(c)  # Uses the default gradient for tf.square.
      with g.gradient_override_map({"Square": "CustomSquare"}):
        s_2 = tf.square(c, name='Square')

      optimizer = tf.train.GradientDescentOptimizer(0.1)#.minimize(loss)
      grad_val = optimizer.compute_gradients(s_2)
      train_op = optimizer.apply_gradients(grad_val)

    with tf.Session(graph=g) as session:
        session.run(tf.global_variables_initializer())

        _, g_v = session.run([train_op, grad_val])

        for g, v in g_v:
            print(g)
            print('*' * 80)
            print(v)
    # [ 101.]
    # ********************************************************************************
    # [-5.10000038]

def test_abs():
    @tf.RegisterGradient("CustomSquare")
    def _custom_square_grad(op, grad):
      return tf.constant([101.0])

    with tf.Graph().as_default() as g:
      c = tf.Variable([5.0], dtype=tf.float32)
      with g.gradient_override_map({"Abs": "CustomSquare"}):
        s_2 = tf.abs(c, name='Abs')

      optimizer = tf.train.GradientDescentOptimizer(0.1)#.minimize(loss)
      grad_val = optimizer.compute_gradients(s_2)
      train_op = optimizer.apply_gradients(grad_val)

    with tf.Session(graph=g) as session:
        session.run(tf.global_variables_initializer())

        _, g_v = session.run([train_op, grad_val])

        for g, v in g_v:
            print(g)
            print('*' * 80)
            print(v)
    # [ 101.]
    # ********************************************************************************
    # [-5.10000038]

if __name__ == "__main__":
    # test_update_norm()
    # test_boolean_index()
    # test_concat_ids()
    test_matmul()
    # test_square()
    # test_abs()
