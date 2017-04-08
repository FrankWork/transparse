import tensorflow as tf
from tensorflow.python import debug as tf_debug
import time
import numpy as np
import os
import sys
import threading
import queue

import data_utils
import transparse_model


# compile:
# TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
# g++ -std=c++11 -shared norm_prjct_op.cc norm_prjct_kernel.cc -o norm_prjct_op.so -fPIC -I $TF_INC -O2 -D_GLIBCXX_USE_CXX11_ABI=0


tf.app.flags.DEFINE_string("data_dir", "data/", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("margin", 4, "Used in margin-based loss function.")
tf.app.flags.DEFINE_integer("relation_num", 11,
                            "Lelation number and sparse degree of matrix.")
tf.app.flags.DEFINE_integer("batch_size", 1000,
                            "Size of the mini-batch.")
tf.app.flags.DEFINE_integer("embedding_size", 20, "embedding_size")
tf.app.flags.DEFINE_integer("epochs", 50,
                            "How many epochs to run.")
tf.app.flags.DEFINE_string("job_name", "", "ps or worker")
tf.app.flags.DEFINE_integer("task_index", "0", "task_index")
tf.app.flags.DEFINE_integer("epochs_per_eval", 20,
                            "How many training epochs write parameters to file.")
tf.app.flags.DEFINE_integer("thread_num", 12,
                            "How many training threads to run.")
tf.app.flags.DEFINE_boolean("use_bern", True,
                            "Bernoulli or uniform distribution.")
tf.app.flags.DEFINE_boolean("l1_norm", True, "L1 Norm.")
tf.app.flags.DEFINE_boolean("test", False,
                            "Set to True for test.")
tf.app.flags.DEFINE_boolean("debug", False,
                            "Set to True for test.")
tf.app.flags.DEFINE_boolean("random_embed", False,
                            "Set to True for random entity embeddings.")
tf.app.flags.DEFINE_boolean("mt", False,
                            "Set to True for multithreading training.")

FLAGS = tf.app.flags.FLAGS
ps_hosts = ['localhost:8088','localhost:8188']
# ps_hosts = ['localhost:8088']
worker_hosts = ['localhost:8288']

# worker_hosts = ['localhost:8288','localhost:8388','localhost:8488']


def create_model():
    data_mgr = data_utils.DataMgr(FLAGS.data_dir, FLAGS.batch_size,
                                  FLAGS.use_bern, FLAGS.embedding_size)
    if FLAGS.random_embed:
        data_mgr.entity_embeddings = None
        data_mgr.relation_embeddings = None
    # NOTE: for debug
    # data_mgr.steps_per_epoch = 1

    model = transparse_model.TranSparseModel(ps_hosts,
                                             worker_hosts,
                                             FLAGS.job_name,
                                             FLAGS.task_index,
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
    # NUM_THREADS = 96
    # config = tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS)
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
    '''
    only use 7 cores on a 24 cores machine!!!
    about 1.5s per epoch
    '''
    data_mgr, model = create_model()
    saver = model.saver
    # Config to turn on JIT compilation
    NUM_THREADS = 24
    config = tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS, inter_op_parallelism_threads=NUM_THREADS)
    # config = tf.ConfigProto()
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
        
        # lock = threading.Lock()
        # result_queue = queue.Queue()
        def _thread_body(coord):
            total_steps = FLAGS.epochs * data_mgr.steps_per_epoch
            total_steps -= total_steps % FLAGS.thread_num
            steps = total_steps // FLAGS.thread_num
            # print('steps: %d' % steps)
            # initial_step, = session.run([model.global_step])
            step_processed = 0
            while not coord.should_stop():
                inputs = data_mgr.get_batch_multi_thread() # it does not improve the performance
                _ = model.train_minibatch_multithread(session, inputs)
                step_processed += 1
                # result_queue.put(summary_and_loss)
                # epoch = (step - initial_step) // data_mgr.steps_per_epoch
                if step_processed >= steps:
                    print('step_processed: %d' % step_processed)
                    # coord.request_stop() # all threads stop!!!!
                    break

        
        initial_step, = session.run([model.global_step])
        # print('initial_step: %d' % initial_step)
        workers = []
        coord = tf.train.Coordinator()
        for i in range(FLAGS.thread_num):
            t = threading.Thread(target=_thread_body, args=(coord, ))
            t.start()
            workers.append(t)
        
        total_steps = FLAGS.epochs * data_mgr.steps_per_epoch
        total_steps -= total_steps % FLAGS.thread_num
        print('total_steps: %d' % total_steps)
        last_step, last_time = initial_step, time.time()
        while True:
            time.sleep(3)

            step, = session.run([model.global_step])
            now = time.time()
            epoch = (step - initial_step) // data_mgr.steps_per_epoch
            if (step - initial_step) >= total_steps:
                break
            
            if (step == last_step):
                break
            time_per_step = (now - last_time) / (step - last_step)
            batch_time = time_per_step * data_mgr.steps_per_epoch 
            print ("epoch %d step %d batch_time %.2f" % (epoch, step,  batch_time))
            sys.stdout.flush()
            last_step, last_time = step, now
            
        coord.join(workers)
        Mh_all, Mt_all, relations, entitys = model.get_matrix_and_embeddings(session)
        write_parameters_to_file(Mh_all, Mt_all, relations, entitys, 0)
        # 12 threads, 20 epochs
        # real	0m38.310s
        # user	2m20.574s
        # sys	0m53.638s
        # 0.859730652504


        # for epoch in range(FLAGS.epochs):
        #     start_time = time.time()
        #     initial_step, = session.run([model.global_step])

            
                

                # if epoch % FLAGS.epochs_per_eval == 0 :
                # Mh_all, Mt_all, relations, entitys = model.get_matrix_and_embeddings(session)
                # write_parameters_to_file(Mh_all, Mt_all, relations, entitys, epoch)
                # # eval(FLAGS.train_dir, epoch)     


                # epoch_time = (time.time() - start_time)
                # # Print statistics for the previous epoch.
                
                # #     train_writer.add_summary(summary, batch_loss)
                # #     epoch_loss += batch_loss

                # print ("epoch %d epoch-time %.2f loss %.2f" % (epoch, epoch_time, epoch_loss))
                # epoch_loss =0
                
                # # Save checkpoint
                # checkpoint_path = os.path.join(FLAGS.train_dir, "transparse.ckpt")
                # saver.save(session, checkpoint_path, global_step=model.global_step)
                # sys.stdout.flush()


            # for t in workers:
            #     t.join() # wait the threads to finish

def train_dist():
    data_mgr, model = create_model()
    # Config to turn on JIT compilation
    NUM_THREADS = 0 # let the system decides
    config = tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=NUM_THREADS, inter_op_parallelism_threads=NUM_THREADS)
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    with tf.train.MonitoredTrainingSession(master=model.server.target,
                                           is_chief=(model.task_index == 0),
                                           checkpoint_dir=FLAGS.train_dir,
                                           save_summaries_steps = None,
                                           config=config#,
                                           #hooks=hooks
                                           ) as session:
        # Run a training step asynchronously.
        # See `tf.train.SyncReplicasOptimizer` for additional details on how to
        # perform *synchronous* training.
        # session.run handles AbortedError in case of preempted PS.
        for op in session.graph.get_operations():
            print('%s\t%s' % (op.name, getattr(op, 'device', 'no device')))
        exit()
        total_steps = FLAGS.epochs * data_mgr.steps_per_epoch
        worker_num = len(worker_hosts)
        total_steps = total_steps // worker_num

        initial_step, = session.run([model.global_step])
        last_time, last_step = time.time(), initial_step
        for local_step in range(total_steps):
            if session.should_stop():
                break
            inputs = data_mgr.get_batch_multi_thread() # it does not improve the performance
            global_step = model.train_minibatch_multithread(session, inputs)
            if global_step % 100 == 0:
                now = time.time()
                epoch = (global_step - initial_step) // data_mgr.steps_per_epoch
                time_per_step = (now - last_time) / (global_step - last_step)
                epoch_time = time_per_step * data_mgr.steps_per_epoch 
                print ("epoch %d global_step %d epoch_time %.2f" % (epoch, global_step, epoch_time))
                sys.stdout.flush()
                last_step, last_time = global_step, now
    
            


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





def main(_):
    train_dist()
    # if FLAGS.dist:
        
    # elif FLAGS.mt:
    #     train_multi_thread()
    # else:
    #     train()

if __name__ == "__main__":
    tf.app.run()
