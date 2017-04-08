import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
import os

op_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'norm_prjct_op.so')
norm_prjct_module = tf.load_op_library(op_path)

class TranSparseModel(object):
    def __init__(self, ps_hosts, worker_hosts, job_name, task_index,
                margin, lr, l1_norm,
                r_num, e_num, # relation_num, entity_num 
                e_sz, b_sz, # embedding_size, batch_size 
                e_embed, r_embed, # pre-trained embeddings 
                mask_h_idx, mask_t_idx):

        # Create a cluster from the parameter server and worker hosts.
        cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

        # Create and start a server for the local task.
        server = tf.train.Server(cluster,
                           job_name=job_name,
                           task_index=task_index)
        self.task_index = task_index
        
        if job_name == "ps":
            server.join()
        elif job_name == "worker":
            # with tf.device("/job:worker/task:%d" % task_index):
            greedy = tf.contrib.training.GreedyLoadBalancingStrategy(len(ps_hosts), tf.contrib.training.byte_size_load_fn)
            with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % task_index, 
                # merge_devices = False,
                cluster=cluster, 
                ps_strategy=greedy
                )):
                
                self._global_tensor(r_num, e_num, e_sz, b_sz, e_embed, r_embed, mask_h_idx, mask_t_idx)
                self.train_op = self._optimize_graph(margin, lr, l1_norm)
                with tf.control_dependencies([self.train_op]):
                    self.norm_op = self._norm()
                    with tf.control_dependencies([self.norm_op]):
                            self.norm_prjct_op = self._norm_projected_cxx()
                            # print(self.Mh_all.device)       # /job:ps/task:0
                            # print(self.Mt_all.device)       # /job:ps/task:1
                            # print(self.relations.device)    # /job:ps/task:0
                            # print(self.entitys.device)      # /job:ps/task:1
                            # print(self.norm_prjct_op.device)# /job:ps/task:0
                            # print(self.train_op.device)     # /job:ps/task:0
                            # print(self.loss.device)         # /job:worker/task:0
                            # print(self.global_step.device)  # /job:ps/task:0
                            # print(self.norm_op.device)      # /job:worker/task:0
                            # exit()

                        # print('-'*40)
                        # print(self.norm_prjct_op)

                self.merged = tf.summary.merge_all()
                self.server = server
                
                # self.saver = tf.train.Saver()

    def _global_tensor(self, r_num, e_num, e_sz, b_sz, e_embed, r_embed, mask_h_idx, mask_t_idx):
        with tf.name_scope("Transfer_Matrix"):
            # Mh is Sparse Transfer Matrix of head for all relations.
            # Mh: r x m x n, m == n, r is relation num, n is embedding_size.
            # Mh[r] is Transfer Matrix for head for r-th relation.
            # Mh[r] is initialized as Identity Matrix.
            # Mt is Sparse Transfer Matrix of tail for all relations
            # The shape of Mt is the same as Mh
            M_init = np.asarray([np.eye(e_sz) for _ in range(r_num)])
            # with tf.device('/job:ps/task:0'):
            Mh_all = tf.Variable(M_init, dtype=tf.float32, name="Mh_all")
            Mt_all = tf.Variable(M_init, dtype=tf.float32, name="Mt_all")
            shape = [r_num, e_sz, e_sz]
            self.mask_h_all = tf.sparse_to_dense(mask_h_idx, shape, tf.ones([len(mask_h_idx)]), name='mask_h_all')
            self.mask_t_all = tf.sparse_to_dense(mask_t_idx, shape, tf.ones([len(mask_t_idx)]), name='mask_t_all')
            self.Mh_all = Mh_all
            self.Mt_all = Mt_all

        with tf.name_scope('Input'):
            # mini-batch input, shape: b, b is batch_size
            rids = tf.placeholder(tf.int32, shape=[b_sz], name='rids')
            hids = tf.placeholder(tf.int32, shape=[b_sz], name='hids')
            tids = tf.placeholder(tf.int32, shape=[b_sz], name='tids')
            n_hids = tf.placeholder(tf.int32, shape=[b_sz], name='n_hids')
            n_tids = tf.placeholder(tf.int32, shape=[b_sz], name='n_tids')
            flag_heads = tf.placeholder(tf.bool, shape=[b_sz], name='flag_heads')
            self.inputs = (rids, hids, tids, n_hids, n_tids, flag_heads)

        with tf.name_scope('Embedding'):
            # relation and entity embedding
            # relation_embeddings shape: r x m
            # entity_embeddings: shape: r x n
            if e_embed is None and r_embed is None:
                # with tf.device('/job:ps/task:0'):
                relations = tf.Variable(tf.truncated_normal([r_num, e_sz, 1]), name='relations')
                entitys = tf.Variable(tf.truncated_normal([e_num, e_sz, 1]), name='entitys')
            else:
                r_embed = tf.reshape(r_embed, [r_num, e_sz, 1])
                e_embed = tf.reshape(e_embed, [e_num, e_sz, 1])
                # with tf.device('/job:ps/task:0'):
                relations = tf.Variable(r_embed, dtype=tf.float32, name='relations')
                entitys = tf.Variable(e_embed, dtype=tf.float32, name='entitys')
            self.relations = relations
            self.entitys = entitys

    def _embed_lookup(self, inputs):
        with tf.name_scope('Embedding_Lookup'):
            (rids, hids, tids, n_hids, n_tids, flag_heads) = inputs
            r = tf.nn.embedding_lookup(self.relations, rids, name='r')
            h = tf.nn.embedding_lookup(self.entitys, hids, name='h')
            t = tf.nn.embedding_lookup(self.entitys, tids, name='t')
            neg_h = tf.nn.embedding_lookup(self.entitys, n_hids, name='neg_h')
            neg_t = tf.nn.embedding_lookup(self.entitys, n_tids, name='neg_t')
            Mh = tf.nn.embedding_lookup(self.Mh_all, rids, name='Mh')
            Mt = tf.nn.embedding_lookup(self.Mt_all, rids, name='Mt')
            mask_h = tf.nn.embedding_lookup(self.mask_h_all, rids, name='mask_h')
            mask_t = tf.nn.embedding_lookup(self.mask_t_all, rids, name='mask_t')
            return r,h,t,neg_h,neg_t,Mh,Mt, mask_h, mask_t

    def _projected_entity(self, Mh, Mt, h, t, neg_h, neg_t):
        with tf.name_scope('Projected_Entity'):
            h_p = tf.matmul(Mh, h, name='h_p')
            t_p = tf.matmul(Mt, t, name='t_p')
            neg_h_p = tf.matmul(Mh, neg_h,  name='neg_h_p')
            neg_t_p = tf.matmul(Mt, neg_t,  name='neg_t_p')
            return h_p, t_p, neg_h_p, neg_t_p

    def _mask_grad(self, Mh_grad, Mt_grad, mask_h, mask_t):
        with tf.name_scope('Mask_Grad'):
            # Mh_grad is IndexedSlices
            # mask_h is Tensor
            mask_gh = tf.IndexedSlices(
                                tf.multiply(Mh_grad.values, mask_h),
                                Mh_grad.indices, Mh_grad.dense_shape
                                )
            mask_gt = tf.IndexedSlices(
                                tf.multiply(Mt_grad.values, mask_t),
                                Mt_grad.indices, Mt_grad.dense_shape
                                )
            return mask_gh, mask_gt

    def _optimize_graph(self, margin, lr, l1_norm):
        with tf.name_scope('Loss'):
            r,h,t,neg_h,neg_t,Mh,Mt,mask_h,mask_t = self._embed_lookup(self.inputs)
            h_p, t_p, neg_h_p, neg_t_p = self._projected_entity(Mh, Mt, h, t, neg_h, neg_t)

            if l1_norm:
                score_pos = tf.abs(h_p + r - t_p)
                score_neg = tf.abs(neg_h_p + r - neg_t_p)
            else:
                score_pos = tf.square(h_p + r - t_p)
                score_neg = tf.square(neg_h_p + r - neg_t_p)
            logits = tf.maximum(margin + score_pos - score_neg, 0, name='logits')
            loss = tf.reduce_sum(logits, name='loss')

            tf.summary.scalar('loss', loss)

        with tf.name_scope('Optimizer'):
            global_step = tf.contrib.framework.get_or_create_global_step()
            # global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.GradientDescentOptimizer(lr, use_locking=True)

            var_list = [self.Mh_all, self.Mt_all, self.relations, self.entitys]
            grad_val = optimizer.compute_gradients(loss, var_list)
            # for grad, var in grad_val:
            #     if grad is not None:
            #         print('-' * 40)
            #         print(var.op.name + '/gradients', grad)
            gh, vh = grad_val[0]# Mh_all
            gt, vt = grad_val[1]# Mt_all
            mask_gh, mask_gt = self._mask_grad(gh, gt, mask_h, mask_t)
            mask_grad_val = [(mask_gh, vh), (mask_gt, vt), grad_val[2], grad_val[3]]
            # NOTE: it took a lot lot lot of times to summary histograms !!!!
            # # Add histograms for trainable variables.
            # for var in tf.trainable_variables():
            #     tf.summary.histogram(var.op.name, var)
            #
            # # Add histograms for gradients.
            # for grad, var in mask_grad_val:
            #     if grad is not None:
            #         tf.summary.histogram(var.op.name + '/gradients', grad)
            train_op = optimizer.apply_gradients(mask_grad_val, global_step)

            self.lr = lr
            self.global_step = global_step
            self.loss = loss
            self.optimizer = optimizer
            return train_op

    def _norm(self):
        def norm_op(embeddings, ids, embed):
            norm = tf.norm(embed, axis=1, keep_dims=True)
            indices = tf.where(tf.greater(norm, 1))
            indices, _ = tf.split(indices, [1, -1], 1)

            gather_embed = tf.gather_nd(embed, indices)
            gather_norm = tf.gather_nd(norm, indices)
            gather_ids = tf.gather_nd(ids, indices)

            normed_embed = gather_embed / gather_norm
            # tprint = tf.Print(indices, [indices], summarize=12)
            # with tf.control_dependencies([tprint]):
            return tf.scatter_update(embeddings, gather_ids, normed_embed)

        with tf.name_scope('Norm'):
            (rids, hids, tids, n_hids, n_tids, flag_heads) = self.inputs
            r,h,t,neg_h,neg_t,Mh,Mt,mask_h,mask_t = self._embed_lookup(self.inputs)

            n_ids = tf.where(flag_heads, n_hids, n_tids)
            neg = tf.where(flag_heads, neg_h, neg_t)

            h_norm_op = norm_op(self.entitys, hids, h)
            t_norm_op = norm_op(self.entitys, tids, t)
            neg_norm_op = norm_op(self.entitys, n_ids, neg)
            r_norm_op = norm_op(self.relations, rids, r)

            with tf.control_dependencies([h_norm_op, t_norm_op, neg_norm_op, r_norm_op]):
                return tf.no_op(name='norm_op')

    
    def _norm_projected_cxx(self):
        '''
        Fast and high accuracy!
        '''
        (rids, hids, tids, n_hids, n_tids, flag_heads) = self.inputs

        # g = tf.get_default_graph()
        # with g.colocate_with(self.norm_op):
        return  norm_prjct_module.norm_prjct_op(self.Mh_all, 
                                                self.Mt_all,
                                                self.relations,
                                                self.entitys,
                                                self.mask_h_all,
                                                self.mask_t_all,
                                                self.lr,
                                                rids, hids, tids, 
                                                n_hids, n_tids, flag_heads
                                                )

    def _norm_projected_cxx_v2(self):
        '''
        Fast and high accuracy!
        '''
        (rids, hids, tids, n_hids, n_tids, flag_heads) = self.inputs
        _,h,t,neg_h,neg_t,Mh,Mt,mask_h,mask_t = self._embed_lookup(self.inputs)
        n_ids = tf.where(flag_heads, n_hids, n_tids)
        neg = tf.where(flag_heads, neg_h, neg_t)
        ? = norm_prjct_module.norm_prjct_op_v2(Mh, Mt, h, t, neg_h, neg_t, flag_heads, mask_h, mask_t, lr)
        return  ?


    def train_minibatch(self, session, inputs):
        rids, hids, tids, n_hids, n_tids, flag_heads = self.inputs
        rids_val, hids_val, tids_val, n_hids_val, n_tids_val, flag_heads_val = inputs
        feed_dict = {rids:rids_val, hids:hids_val, tids:tids_val,
                     n_hids:n_hids_val, n_tids:n_tids_val,
                     flag_heads:flag_heads_val}

        summary, _, _, _, loss_val = session.run([self.merged,
                     self.train_op, self.norm_op, self.norm_prjct_op, self.loss],
                     feed_dict=feed_dict)
        
        return summary, loss_val
    def train_minibatch_multithread(self, session, inputs):
        rids, hids, tids, n_hids, n_tids, flag_heads = self.inputs
        rids_val, hids_val, tids_val, n_hids_val, n_tids_val, flag_heads_val = inputs
        feed_dict = {rids:rids_val, hids:hids_val, tids:tids_val,
                     n_hids:n_hids_val, n_tids:n_tids_val,
                     flag_heads:flag_heads_val}

        _, step, _, _ = session.run([
                     self.train_op, self.global_step, self.norm_op, self.norm_prjct_op],
                     feed_dict=feed_dict)
        
        return step
    def get_matrix_and_embeddings(self, session):
        return session.run([self.Mh_all, self.Mt_all, self.relations, self.entitys])
