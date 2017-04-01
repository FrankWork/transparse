import tensorflow as tf
from tensorflow.python import debug as tf_debug
import time
import numpy as np
import os
import sys

import data_utils
import transparse_model


# compile:
# TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
# g++ -std=c++11 -shared norm_prjct_op.cc norm_prjct_kernel.cc -o norm_prjct_op.so \
#     -fPIC -I $TF_INC -O2 -D_GLIBCXX_USE_CXX11_ABI=0


tf.app.flags.DEFINE_string("data_dir", "data/", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "output/", "Training directory.")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("margin", 4, "Used in margin-based loss function.")
tf.app.flags.DEFINE_integer("relation_num", 11,
                            "Lelation number and sparse degree of matrix.")
tf.app.flags.DEFINE_integer("epochs", 1,
                            "How many epochs to run.")
tf.app.flags.DEFINE_integer("epochs_per_eval", 1,
                            "How many training epochs write parameters to file.")
tf.app.flags.DEFINE_integer("batch_size", 100,
                            "Size of the mini-batch.")
tf.app.flags.DEFINE_integer("embedding_size", 20, "embedding_size")
tf.app.flags.DEFINE_boolean("use_bern", True,
                            "Bernoulli or uniform distribution.")
tf.app.flags.DEFINE_boolean("l1_norm", True, "L1 Norm.")
tf.app.flags.DEFINE_boolean("test", False,
                            "Set to True for test.")
tf.app.flags.DEFINE_boolean("debug", False,
                            "Set to True for test.")
tf.app.flags.DEFINE_boolean("random_embed", False,
                            "Set to True for random entity embeddings.")
tf.app.flags.DEFINE_boolean("multi_thread", False,
                            "Set to True for multithreading training.")

FLAGS = tf.app.flags.FLAGS

def create_model():
    data_mgr = data_utils.DataMgr(FLAGS.data_dir, FLAGS.batch_size,
                                  FLAGS.use_bern, FLAGS.embedding_size)
    if FLAGS.random_embed:
        data_mgr.entity_embeddings = None
        data_mgr.relation_embeddings = None
    model = transparse_model.TranSparseModel(
                                             FLAGS.margin,
                                             FLAGS.learning_rate,
                                             FLAGS.l1_norm,
                                             data_mgr.relation_num,
                                             data_mgr.entity_num,
                                             FLAGS.embedding_size,
                                             FLAGS.batch_size,
                                             data_mgr.entity_embeddings,
                                             data_mgr.relation_embeddings,
                                             data_mgr.sparse_index_head,
                                             data_mgr.sparse_index_tail
                                             )
    print('*' * 80)
    print('Parameters:')
    print('\tdata_path: %s' % FLAGS.data_dir)
    print('\tuse_bern: %s' % 'True' if FLAGS.use_bern else 'False')
    print('\tmargin: %d' % FLAGS.margin)
    print('\tlearning_rate: %s' % FLAGS.learning_rate)
    print('\tl1_norm: %d' % FLAGS.l1_norm)
    print('\trandom_embed: %d' % FLAGS.random_embed)

    print('\tepochs: %d' % FLAGS.epochs)
    print('\tsteps_per_epoch: %d' % data_mgr.steps_per_epoch)
    print('\tbatch_size: %d' % FLAGS.batch_size)
    print('\tembedding_size: %d' % FLAGS.embedding_size)
    print('\trelation_num: %d' % data_mgr.relation_num)
    print('\tentity_num: %d' % data_mgr.entity_num)
    print('*' * 80)
    return data_mgr, model

def train():
    data_mgr, model = create_model()
    saver = model.saver
    # Config to turn on JIT compilation
    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    with tf.Session(config=config) as session:
        if FLAGS.debug:
            session = tf_debug.LocalCLIDebugWrapperSession(session)
            session.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        print('*'*40)
        train_writer = tf.summary.FileWriter(FLAGS.train_dir, session.graph)

        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
          print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
          saver.restore(session, ckpt.model_checkpoint_path)
        else:
          print("Created model with fresh parameters.")
          session.run(tf.global_variables_initializer())

        for epoch in range(FLAGS.epochs):
            if epoch % FLAGS.epochs_per_eval == 0 :
                Mh_all, Mt_all, relations, entitys = model.get_matrix_and_embeddings(session)
                write_parameters_to_file(Mh_all, Mt_all, relations, entitys, epoch)
                # eval(FLAGS.train_dir, epoch)

            start_time = time.time()
            epoch_loss = 0
            for _ in range(data_mgr.steps_per_epoch):
                inputs = data_mgr.get_batch()
                summary, batch_loss = model.train_minibatch(session, inputs)
                train_writer.add_summary(summary, batch_loss)
                epoch_loss += batch_loss

            epoch_time = (time.time() - start_time)
            # Print statistics for the previous epoch.
            print ("epoch %d epoch-time %.2f loss %.2f" % (epoch, epoch_time, epoch_loss))
            # Save checkpoint
            checkpoint_path = os.path.join(FLAGS.train_dir, "transparse.ckpt")
            saver.save(session, checkpoint_path, global_step=model.global_step)
            sys.stdout.flush()

def train_multi_thread():
    data_mgr, model = create_model()
    saver = model.saver
    # Config to turn on JIT compilation
    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    with tf.Session(config=config) as session:
        print('*'*40)
        train_writer = tf.summary.FileWriter(FLAGS.train_dir, session.graph)
        # test_writer = tf.summary.FileWriter(FLAGS.train_dir + '/summaries/test')

        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
          print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
          saver.restore(session, ckpt.model_checkpoint_path)
        else:
          print("Created model with fresh parameters.")
          session.run(tf.global_variables_initializer())

        # for epoch in range(FLAGS.epoch):
        #     start_time = time.time()
        #     epoch_loss = 0
        #     # num_batches is how many steps in one epoch to train
        #     for _ in range(data_mgr.num_batches):
        #         rids, hids, tids, n_hids, n_tids, flag_heads = data_mgr.get_batch()
        #         summary, batch_loss = model.train_minibatch(session, rids, hids,
        #                                     tids, n_hids, n_tids, flag_heads)
        #         # print(batch_loss)
        #         train_writer.add_summary(summary, batch_loss)
        #         epoch_loss += batch_loss# / data_mgr.num_batches
        #
        #     epoch_time = (time.time() - start_time)
        #     # Print statistics for the previous epoch.
        #     print ("epoch %d epoch-time %.2f loss %.2f" % (epoch,
        #                      epoch_time, epoch_loss))
        #     # Save checkpoint
        #     checkpoint_path = os.path.join(FLAGS.train_dir, "transparse.ckpt")
        #     saver.save(session, checkpoint_path, global_step=model.global_step)
        #     sys.stdout.flush()

def write_parameters_to_file(Mh, Mt, relations, entitys, epoch):
    """
    the type of Mh, Mt, relations, entitys is np.ndarray
    """
    path = FLAGS.train_dir
    relations_file = os.path.join(path,
                                'relation2vec.bern' + str(epoch))
    relation_num, embedding_size, _ = relations.shape
    relations = relations.reshape((relation_num, embedding_size))
    with open(relations_file, 'w') as f:
        for row in range(relation_num):
            for col in range(embedding_size):
                f.write('%.6f ' % relations[row][col])
            f.write('\n')


    entitys_file = os.path.join(path,
                                'entity2vec.bern' + str(epoch))
    entity_num, embedding_size, _ = entitys.shape
    entitys = entitys.reshape((entity_num, embedding_size))
    with open(entitys_file, 'w') as f:
        for row in range(entity_num):
            for col in range(embedding_size):
                f.write('%.6f ' % entitys[row][col])
            f.write('\n')

    A_h_file = os.path.join(path,
                                'A_h.bern' + str(epoch))
    relation_num, rows, cols = Mh.shape
    with open(A_h_file, 'w') as f:
        for r in range(relation_num):
            for row in range(rows):
                for col in range(cols):
                    f.write('%.6f ' % Mh[r][row][col])
                f.write('\n')

    A_t_file = os.path.join(path,
                                'A_t.bern' + str(epoch))
    relation_num, rows, cols = Mt.shape
    with open(A_t_file, 'w') as f:
        for r in range(relation_num):
            for row in range(rows):
                for col in range(cols):
                    f.write('%.6f ' % Mt[r][row][col])
                f.write('\n')

# def write_parameters_for_eval():
#     path = FLAGS.train_dir
#
#     data_mgr, model = create_model()
#     saver = model.saver
#
#     with tf.Session() as session:
#         print("Created model with fresh parameters.")
#         session.run(tf.global_variables_initializer())
#         Mh, Mt, relations, entitys = model.get_matrix_and_embeddings(session)
#         write_parameters_to_file(Mh, Mt, relations, entitys, 0)
#
#         ckpt = tf.train.get_checkpoint_state(path)
#         if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
#             print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
#             saver.restore(session, ckpt.model_checkpoint_path)
#             Mh, Mt, relations, entitys = model.get_matrix_and_embeddings(session)
#             write_parameters_to_file(Mh, Mt, relations, entitys, 999)

def test_small_model():
    relation_num = 2
    entity_num = 4
    embedding_size = 2
    batch_size = 1
    epoch_num = 10
    lr = 0.001
    margin = 4

    rids = [0]
    hids = [0]
    tids = [1]
    n_hids = [0]
    n_tids = [2]
    flag_heads = [False]
    index_h = [[0,0,0], [0,1,1], [1,0,0], [1,1,1]]
    index_t = [[0,0,0], [0,1,1], [1,0,0], [1,1,1]]
    inputs = (rids, hids, tids, n_hids, n_tids, flag_heads)

    model = transparse_model.TranSparseModel(
                                             FLAGS.margin,
                                             FLAGS.learning_rate,
                                             FLAGS.l1_norm,
                                             relation_num,
                                             entity_num,
                                             embedding_size,
                                             batch_size,
                                             None,
                                             None,
                                             index_h,
                                             index_t
                                             )


    with tf.Session() as session:
        print('*'*40)
        if FLAGS.debug:
            session = tf_debug.LocalCLIDebugWrapperSession(session)
            session.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        session.run(tf.global_variables_initializer())

        for epoch in range(epoch_num):
            Mh_all, Mt_all, relations, entitys = model.get_matrix_and_embeddings(session)
            write_parameters_to_file(Mh_all, Mt_all, relations, entitys, epoch)
            _, batch_loss = model.train_minibatch(session, inputs)
            print ("epoch %d loss %.2f" % (epoch, batch_loss))

def main(_):
    if FLAGS.test:
        test_small_model()
    # elif FLAGS.eval:
    #     print('write_parameters_for_eval:')
    #     write_parameters_for_eval()
    elif FLAGS.multi_thread:
        train_multi_thread()
    else:
        train()

if __name__ == "__main__":
    tf.app.run()
