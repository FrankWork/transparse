import tensorflow as tf

import data_utils
import transparse_model
import time
import numpy as np
import os
import sys

tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "model-output", "Training directory.")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("margin", 4, "Used in margin-based loss function.")
tf.app.flags.DEFINE_integer("relation_num", 11,
                            "Lelation number and sparse degree of matrix.")
tf.app.flags.DEFINE_integer("epoch", 100,# 1000
                            "Epoch number to run.")
tf.app.flags.DEFINE_integer("num_batches", 100,
                            "Number of batches to generate in one epoch.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 20,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("n", 20,
                            "size???????????????????????????.")
tf.app.flags.DEFINE_boolean("use_bern", True,
                            "Bernoulli or uniform distribution.")
tf.app.flags.DEFINE_boolean("train", True,
                            "Set to True for traing.")
tf.app.flags.DEFINE_boolean("use_fp16", False,
                            "Train using fp16 instead of fp32.")

FLAGS = tf.app.flags.FLAGS

def create_model(session):
    """Create TranSparseModel and initialize or load parameters in session."""
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32

    # FIXME:
    model = transparse_model.TranSparseModel()

    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
      print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
      model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
      print("Created model with fresh parameters.")
      session.run(tf.global_variables_initializer())

    tf.summary.scalar("Loss", model.loss)

    return model

def main1(_):
  # Prepare training data.
  print("Preparing training data in %s" % FLAGS.data_dir)
  # data_mgr = data_utils.DataMgr(FLAGS.data_dir, FLAGS.batch_size, FLAGS.use_bern)

  with tf.Session() as sess:
      model = create_model(sess)

      # training loop
      step_time, loss = 0.0, 0.0
      current_step = 0
      for _ in range(FLAGS.epoch):
          # get a batch and make a step
          start_time = time.time()
          # FIXME:
          batch_data = data_mgr.get_batch()
          step_loss = model.step()

          step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
          # FIXME: ?
          loss += step_loss / FLAGS.steps_per_checkpoint
          current_step += 1

          # Once in a while, we save checkpoint, print statistics, and run evals.
          if current_step % FLAGS.steps_per_checkpoint == 0:
              # Print statistics for the previous epoch.
              # FIXME
              print ("global step %d learning rate %.4f step-time %.2f perplexity "
                     "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                               step_time, loss))
              # Decrease learning rate
              pass
              # Save checkpoint and zero timer and loss.
              checkpoint_path = os.path.join(FLAGS.train_dir, "transparse.ckpt")
              # FIXME
              model.saver.save(sess, checkpoint_path, global_step=model.global_step)
              step_time, loss = 0.0, 0.0
              # Run evals on development set
              pass
              sys.stdout.flush()

def main(_):
    model = transparse_model.TranSparseModel()
    saver = model.saver
    with tf.Session() as session:
        print('*'*40)
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

        step_time = 0
        for epoch in range(FLAGS.epoch):
            start_time = time.time()

            # summery, _, loss_val = session.run([merged, train, loss],
            #         {x_train:[1, 2, 3, 4], y_train:[0, 1, 2, 3]})
            summery, loss_val = model.step(session, [1, 2, 3, 4], [0, 1, 2, 3])

            step_time += (time.time() - start_time)

            # Print statistics for the previous epoch.
            if epoch % 10 == 0:
                train_writer.add_summary(summery, loss_val)
            if epoch % FLAGS.steps_per_checkpoint == 0:
                print ("epoch %d step-time %.2f loss %.2f" % (epoch,
                                 step_time, loss_val))
                step_time = 0

                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
                saver.save(session, checkpoint_path, global_step=model.global_step)
                sys.stdout.flush()
        checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
        saver.save(session, checkpoint_path, global_step=model.global_step)

if __name__ == "__main__":
  tf.app.run()
