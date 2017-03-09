import tensorflow as tf

import data_utils
import transparse_model
import time
import numpy as np
import os
import sys

tf.app.flags.DEFINE_string("data_dir", "data/", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "model-output/", "Training directory.")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("margin", 4, "Used in margin-based loss function.")
tf.app.flags.DEFINE_integer("relation_num", 11,
                            "Lelation number and sparse degree of matrix.")
tf.app.flags.DEFINE_integer("epoch", 1#100,# 1000
                            "Epoch number to run.")
tf.app.flags.DEFINE_integer("batch_size", 100,
                            "Size of the mini-batch.")
# tf.app.flags.DEFINE_integer("steps_per_checkpoint", 20,
#                             "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("embedding_size", 20, "embedding_size")
tf.app.flags.DEFINE_boolean("use_bern", True,
                            "Bernoulli or uniform distribution.")
tf.app.flags.DEFINE_boolean("train", True,
                            "Set to True for traing.")
tf.app.flags.DEFINE_boolean("use_fp16", False,
                            "Train using fp16 instead of fp32.")
tf.app.flags.DEFINE_boolean("l1_norm", False,
                            "L1 norm.")

FLAGS = tf.app.flags.FLAGS


def main(_):
    print('*'*40)
    data_mgr = data_utils.DataMgr(FLAGS.data_dir, FLAGS.batch_size,
                                  FLAGS.use_bern, FLAGS.embedding_size)
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    model = transparse_model.TranSparseModel(data_mgr.relation_num,
                                             data_mgr.entity_num,
                                             data_mgr.embedding_size,
                                             data_mgr.sparse_index_head,
                                             data_mgr.sparse_index_tail,
                                             FLAGS.learning_rate, dtype,
                                             FLAGS.l1_norm)
    saver = model.saver
    with tf.Session() as session:
        train_writer = tf.summary.FileWriter(FLAGS.train_dir + '/summaries/train',
                                              session.graph)
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
                batch_data = data_mgr.get_batch()
                summary, batch_loss = model.step(session, batch_data)
                epoch_loss += batch_loss
                train_writer.add_summary(summary, batch_loss)

            epoch_time = (time.time() - start_time)
            # Print statistics for the previous epoch.
            print ("epoch %d step-time %.2f loss %.2f" % (epoch,
                             epoch_time, epoch_loss))
            # Save checkpoint and zero timer and loss.
            checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
            saver.save(session, checkpoint_path, global_step=model.global_step)
            sys.stdout.flush()

if __name__ == "__main__":
    tf.app.run()
