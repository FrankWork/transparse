import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np

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

def test_multiply_IndexedSlices():
    with tf.Graph().as_default() as g:
        values_a = tf.Variable(tf.reshape(tf.range(9, dtype=tf.float32), [3,3]) )
        values_b = tf.Variable(tf.reshape(tf.range(9, dtype=tf.float32), [3,3]) )
        index_a = tf.constant([0, 1, 3])
        index_b = tf.constant([0, 1, 3])
        dense_shape = tf.constant([10, 3])
        indexed_a = tf.IndexedSlices(values_a, index_a, dense_shape)
        indexed_b = tf.IndexedSlices(values_b, index_b, dense_shape)

        tensor_c = tf.multiply(indexed_a, indexed_b)

        index_c = indexed_a.indices
        dense_shape_c = indexed_a.dense_shape
        values_c = tf.multiply(indexed_a.values, indexed_b.values)
        indexed_c = tf.IndexedSlices(values_c, index_c, dense_shape_c)
        # print(type(c))
        # d = tf.constant(c)

    with tf.Session(graph=g) as session:
        session.run(tf.global_variables_initializer())

        print(session.run(values_a))
        print(session.run(index_a))
        print(session.run(indexed_a))
        print(session.run(tensor_c))
        print(session.run(indexed_c))



if __name__ == "__main__":
    # test_update_norm()
    # test_boolean_index()
    # test_concat_ids()
    # test_matmul()
    # test_square()
    test_multiply_IndexedSlices()
