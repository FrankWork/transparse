import tensorflow as tf
from tensorflow.python import debug as tf_debug
import time
import numpy as np
import os
import sys

import data_utils
import transparse_model

tf.app.flags.DEFINE_string("data_dir", "data/", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "output/", "Training directory.")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("margin", 4, "Used in margin-based loss function.")
tf.app.flags.DEFINE_integer("relation_num", 11,
                            "Lelation number and sparse degree of matrix.")
tf.app.flags.DEFINE_integer("epoch", 50,
                            "Epoch number to run.")
tf.app.flags.DEFINE_integer("batch_size", 100,
                            "Size of the mini-batch.")
# tf.app.flags.DEFINE_integer("steps_per_checkpoint", 20,
#                             "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("embedding_size", 20, "embedding_size")
tf.app.flags.DEFINE_boolean("use_bern", True,
                            "Bernoulli or uniform distribution.")
tf.app.flags.DEFINE_boolean("l1_norm", True, "L1 Norm.")
tf.app.flags.DEFINE_boolean("test", False,
                            "Set to True for test.")
tf.app.flags.DEFINE_boolean("debug", False,
                            "Set to True for test.")
tf.app.flags.DEFINE_boolean("eval", False,
                            "Set to True for eval.")

FLAGS = tf.app.flags.FLAGS

def train():
    print('*'*40)
    data_mgr = data_utils.DataMgr(FLAGS.data_dir, FLAGS.batch_size,
                                  FLAGS.use_bern, FLAGS.embedding_size)
    model = transparse_model.TranSparseModel(
                                             FLAGS.margin,
                                             FLAGS.learning_rate,
                                             FLAGS.l1_norm,
                                             data_mgr.relation_num,
                                             data_mgr.entity_num,
                                             data_mgr.embedding_size,
                                             data_mgr.batch_size,
                                             data_mgr.entity_embeddings,
                                             data_mgr.relation_embeddings,
                                             data_mgr.sparse_index_head,
                                             data_mgr.sparse_index_tail
                                             )
    saver = model.saver
    with tf.Session() as session:
        train_writer = tf.summary.FileWriter(FLAGS.train_dir, session.graph)
        # test_writer = tf.summary.FileWriter(FLAGS.train_dir + '/summaries/test')

        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
          print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
          saver.restore(session, ckpt.model_checkpoint_path)
        else:
          print("Created model with fresh parameters.")
          session.run(tf.global_variables_initializer())

        for epoch in range(FLAGS.epoch):
            start_time = time.time()
            epoch_loss = 0
            for _ in range(data_mgr.num_batches):
                rids, hids, tids, n_hids, n_tids, flag_heads = data_mgr.get_batch()
                summary, batch_loss = model.train_minibatch(session, rids, hids,
                                            tids, n_hids, n_tids, flag_heads)
                # print(batch_loss)
                train_writer.add_summary(summary, batch_loss)
                epoch_loss += batch_loss# / data_mgr.num_batches

            epoch_time = (time.time() - start_time)
            # Print statistics for the previous epoch.
            print ("epoch %d epoch-time %.2f loss %.2f" % (epoch,
                             epoch_time, epoch_loss))
            # Save checkpoint
            checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
            saver.save(session, checkpoint_path, global_step=model.global_step)
            sys.stdout.flush()

def write_parameters_for_eval(path):
    print('*'*40)
    if path is None:
        path = FLAGS.train_dir

    data_mgr = data_utils.DataMgr(FLAGS.data_dir, FLAGS.batch_size,
                                  FLAGS.use_bern, FLAGS.embedding_size)
    model = transparse_model.TranSparseModel(data_mgr.relation_num,
                                             data_mgr.entity_num,
                                             data_mgr.embedding_size,
                                             data_mgr.batch_size,
                                             data_mgr.entity_embeddings,
                                             data_mgr.relation_embeddings,
                                            #  data_mgr.sparse_index_head,
                                            #  data_mgr.sparse_index_tail,
                                             FLAGS.margin,
                                             FLAGS.learning_rate
                                             )
    saver = model.saver

    def write_parameters_to_file(Mh, Mt, relations, entitys, epoch):
        """
        the type of Mh, Mt, relations, entitys is np.ndarray
        """
        relations_file = path + 'relation2vec.bern' + str(epoch)
        relation_num, embedding_size, _ = relations.shape
        relations = relations.reshape((relation_num, embedding_size))
        with open(relations_file, 'w') as f:
            for row in range(relation_num):
                for col in range(embedding_size):
                    f.write('%.6f ' % relations[row][col])
                f.write('\n')


        entitys_file = path + 'entity2vec.bern' + str(epoch)
        entity_num, embedding_size, _ = entitys.shape
        entitys = entitys.reshape((entity_num, embedding_size))
        with open(entitys_file, 'w') as f:
            for row in range(entity_num):
                for col in range(embedding_size):
                    f.write('%.6f ' % entitys[row][col])
                f.write('\n')

        A_h_file = path + 'A_h.bern' + str(epoch)
        relation_num, rows, cols = Mh.shape
        with open(A_h_file, 'w') as f:
            for r in range(relation_num):
                for row in range(rows):
                    for col in range(cols):
                        f.write('%.6f ' % Mh[r][row][col])
                    f.write('\n')

        A_t_file = path + 'A_t.bern' + str(epoch)
        relation_num, rows, cols = Mt.shape
        with open(A_t_file, 'w') as f:
            for r in range(relation_num):
                for row in range(rows):
                    for col in range(cols):
                        f.write('%.6f ' % Mt[r][row][col])
                    f.write('\n')


    with tf.Session() as session:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        Mh, Mt, relations, entitys = model.get_matrix_and_embeddings(session)
        write_parameters_to_file(Mh, Mt, relations, entitys, 0)

        ckpt = tf.train.get_checkpoint_state(path)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            saver.restore(session, ckpt.model_checkpoint_path)
            Mh, Mt, relations, entitys = model.get_matrix_and_embeddings(session)
            write_parameters_to_file(Mh, Mt, relations, entitys, 999)

def test_small_model():
    relation_num = 2
    entity_num = 4
    embedding_size = 2
    batch_size = 1
    epoch_num = 1
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

        # Mh, Mt, relations, entitys = model.get_matrix_and_embeddings(session)
        # print('*' * 80)
        # print('Mh')
        # print(Mh)
        # print('*' * 80)
        # print('Mt')
        # print(Mt)
        # print('*' * 80)
        # print('relations')
        # print(relations)
        # print('*' * 80)
        # print('entitys')
        # print(entitys)
        # print('*' * 80)
        # mask_h, mask_t = model.get_mask(session)
        # print('*' * 80)
        # print('mask_h')
        # print(mask_h)
        # print('*' * 80)
        # print('mask_t')
        # print(mask_t)

        # for i in range(epoch_num):
        #     logits, loss, Mh, Mt, r, e, grad_val = model.test_minibatch(
        #                 session, rids, hids, tids, n_hids, n_tids, flag_heads)
        #     print('step: %d, loss: %.2f' % (i, loss))
        #
        # gMh, _ = grad_val[0]
        # gMt, _ = grad_val[1]
        # gr, _ = grad_val[2]
        # ge, _ = grad_val[3]
        #
        # print('='*80)
        # print('='*80)
        #
        # print('logits: ')
        # print(logits)
        # print('*' * 80)
        # print('loss: ')
        # print(loss)
        # print('*' * 80)
        # print('Mh')
        # print(Mh)
        # print('*' * 80)
        # print('gMh')
        # print(gMh)
        # print('*' * 80)
        # print('Mt')
        # print(Mt)
        # print('*' * 80)
        # print('gMt')
        # print(gMt)
        # print('*' * 80)
        # print('r')
        # print(r)
        # print('*' * 80)
        # print('gr')
        # print(gr)
        # print('*' * 80)
        # print('e')
        # print(e)
        # print('*' * 80)
        # print('ge')
        # print(ge)

        for i in range(epoch_num):
            model.test_minibatch(session, rids, hids, tids, n_hids, n_tids, flag_heads)
            # print('step: %d, loss: %.2f' % (i, loss))

def main(_):
    if FLAGS.test:
        test_small_model()
    elif FLAGS.eval:
        print('eval:')
        path = '1000-epoch-norm-p-output-3-20/'
        write_parameters_for_eval(path)
    else:
        train()

if __name__ == "__main__":
    tf.app.run()
