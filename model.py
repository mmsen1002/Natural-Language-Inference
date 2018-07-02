# coding=utf-8

import sys
import tensorflow as tf
import logging.config
import config
from nn import highway
from nn import self_attention
from nn import esim_match
from nn import mlp
from utils import read_dataset
from utils import evaluation_rate


class Model:
    def __init__(self, embed_matrix_init, grad_clip,
                 is_train=True, is_continue=False):

        self.embed_matrix_init = embed_matrix_init
        self.grad_clip = grad_clip
        self.is_continue = is_continue
        self.is_train = is_train
        self.global_step = tf.Variable(
            0, dtype=tf.int32, trainable=False, name='global_step')
        self.global_epoch = tf.Variable(
            0, dtype=tf.int32, trainable=False, name='global_epoch')
        self.MODEL_FILE = './save_model/'

    def _get_data(self):

        # read train_dataset
        # train_folder = './data/ccks/train/'
        train_folder = './data/atec/train/'
        train_data, valid_data = read_dataset(
            passage_filename=train_folder+'passage.pkl',
            passage_len_filename=train_folder+'passage_length.pkl',
            query_filename=train_folder+'query.pkl',
            query_len_filename=train_folder+'query_length.pkl',
            label_filename=train_folder+'label.pkl',
            batch_size=config.batch_size)
        """
        # read test_dataset
        test_folder = './data/ccks/dev/'
        #test_folder = './data/ccks/extra_eval/'
        test_data = read_dataset(
            passage_filename=test_folder+'passage.pkl',
            passage_len_filename=test_folder+'passage_length.pkl',
            query_filename=test_folder+'query.pkl',
            query_len_filename=test_folder+'query_length.pkl',
            label_filename=test_folder+'label.pkl',
            batch_size=config.batch_size,
            is_train=False)
        """
        iterator = tf.data.Iterator.from_structure(
            train_data.output_types, train_data.output_shapes)

        self.passage, self.passage_length, self.query,\
            self.query_length, self.labels = iterator.get_next()

        self.train_init = iterator.make_initializer(train_data)
        #self.test_init = iterator.make_initializer(test_data)
        self.valid_init = iterator.make_initializer(valid_data)

        print("_get_data function worked")

    def _create_embedding(self):
        with tf.variable_scope('embedding', reuse=tf.AUTO_REUSE):
            # init embedding_matrix
            self.word_embeddings = tf.get_variable(
                name='word_embedding',
                shape=[config.word_vocab_size, config.word_embed_size],
                dtype=tf.float32,
                #initializer=tf.random_uniform_initializer(-1.0, 1.0, seed=113),
                initializer=tf.constant_initializer(self.embed_matrix_init),
                trainable=True)

            #print("shape of passage == ", self.passage.shape.as_list())
            #print("shape of query == ", self.query.shape.as_list())

            # embedding_lookup
            self.passage_embed = tf.nn.embedding_lookup(
                self.word_embeddings, self.passage)
            self.query_embed = tf.nn.embedding_lookup(
                self.word_embeddings, self.query)
            # dropout
            if self.is_train:
                self.passage_embed = tf.nn.dropout(
                    self.passage_embed, config.keep_prob)
                self.query_embed = tf.nn.dropout(
                    self.query_embed, config.keep_prob)

            # highway network layer
            self.passage_embed = highway(
                self.passage_embed,
                #size = self.rnn_hidden_units,
                scope="highway",
                keep_prob=config.keep_prob,
                is_train=self.is_train,
                reuse=None)
            self.query_embed = highway(
                self.query_embed,
                #size = self.rnn_hidden_units,
                scope="highway",
                keep_prob=config.keep_prob,
                is_train=self.is_train,
                reuse=True)

        print("_create_embedding and highway layer worked")

    def _create_bgrucell(self):
        with tf.variable_scope("bgru_layer"):
            cell_fw = tf.contrib.rnn.GRUCell(
                num_units=config.rnn_cell_size,
                #kernel_initializer=tf.orthogonal_initializer(seed=113))
                kernel_initializer=tf.truncated_normal_initializer(
                    mean=0.0, stddev=0.1, seed=113)
                )
            cell_bw = tf.contrib.rnn.GRUCell(
                num_units=config.rnn_cell_size,
                #kernel_initializer=tf.orthogonal_initializer(seed=114))
                kernel_initializer=tf.truncated_normal_initializer(
                    mean=0.0, stddev=0.1, seed=114)
                )

            if self.is_train:
                cell_fw = tf.contrib.rnn.DropoutWrapper(
                    cell_fw,
                    input_keep_prob=config.keep_prob,
                    output_keep_prob=config.keep_prob)
                cell_bw = tf.contrib.rnn.DropoutWrapper(
                    cell_bw,
                    input_keep_prob=config.keep_prob,
                    output_keep_prob=config.keep_prob)

        print("_create_bgrucell function worked")
        return cell_fw, cell_bw

    def _create_blstm_cell(self):
        cell_fw = tf.contrib.rnn.LSTMCell(config.rnn_cell_size)
        cell_bw = tf.contrib.rnn.LSTMCell(config.rnn_cell_size)
        if self.is_train:
            cell_fw = tf.contrib.rnn.DropoutWrapper(
                cell_fw,
                input_keep_prob=config.keep_prob,
                output_keep_prob=config.keep_prob)
            cell_bw = tf.contrib.rnn.DropoutWrapper(
                cell_bw,
                input_keep_prob=config.keep_prob,
                output_keep_prob=config.keep_prob)

        print("_create_blstm_cell function worked")
        return cell_fw, cell_bw

    def _create_forward(self):
        # single layer bgru encoder
        with tf.variable_scope('bigru', reuse=tf.AUTO_REUSE):
            cell_fw, cell_bw = self._create_bgrucell()
            #cell_fw, cell_bw = self._create_blstm_cell()
            self.p_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.passage_embed,
                dtype=tf.float32,
                sequence_length=self.passage_length)
            self.q_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.query_embed,
                dtype=tf.float32,
                sequence_length=self.query_length)

            self.p_outputs = tf.concat(self.p_outputs, axis=2)
            self.q_outputs = tf.concat(self.q_outputs, axis=2)
            """
            if self.is_train:
                self.p_outputs = tf.nn.dropout(
                    self.p_outputs, keep_prob=config.keep_prob)
                self.q_outputs = tf.nn.dropout(
                    self.q_outputs, keep_prob=config.keep_prob)
            """
            # print("shape of self.p_outputs", self.p_outputs.shape.as_list())

        print("bgru encoder layer worked")
        """
        # self-attention
        with tf.variable_scope('self_ATT', reuse=tf.AUTO_REUSE):
            self.passage_vec, _ = self_attention(
                inputs=config.p_outputs,
                da=config.att_da, r=config.att_r,
                batch_size=config.batch_size,
                reuse=tf.AUTO_REUSE)

            self.query_vec, _ = self_attention(
                inputs=self.q_outputs,
                da=config.att_da, r=config.att_r,
                batch_size=config.batch_size,
                reuse=tf.AUTO_REUSE)

        print("self-attention layer worked")
        """
        with tf.variable_scope('Match', reuse=tf.AUTO_REUSE):
            self.match_outputs = esim_match(
                p_encoded=self.p_outputs,
                q_encoded=self.q_outputs,
                p_len=self.passage_length,
                q_len=self.query_length,
                max_seq_len=config.max_seq_len,
                keep_prob=config.keep_prob3,
                rnn_cell_size=config.rnn_cell_size,
                is_train=self.is_train)

            print("match_outputs: ", self.match_outputs.shape.as_list())
        # Multi Layer Perceptron
        with tf.variable_scope('MLP', reuse=tf.AUTO_REUSE):
            self.logits = mlp(
                mlp_inputs=self.match_outputs,
                mlp_hidden_size=config.mlp_hidden_size,
                num_class=config.num_class,
                keep_prob=config.keep_prob,
                is_train=self.is_train)
        print("MLP layer worked")

    def _create_prediction(self):
        with tf.name_scope("prediction"):
            prob = tf.nn.softmax(self.logits)
            self.predictions = tf.argmax(prob, 1)
            #print("self.predictions: ", self.predictions.shape.as_list())
            #print("self.labels: ", self.labels.shape.as_list())
            #correct = tf.nn.in_top_k(self.logits, self.labels, 1)
            correct = tf.equal(self.predictions, self.labels)
            self.num_correct = tf.reduce_sum(tf.cast(correct, tf.int32))

    def _create_loss(self):
        with tf.name_scope("loss"):
            gold_matrix = tf.one_hot(
                self.labels, config.num_class, dtype=tf.float32)
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.logits, labels=gold_matrix)
            )

    def _create_optimizer(self):
        with tf.name_scope("optimizer"):
            train_variables = tf.trainable_variables()

            # L2 norm
            if config.lambda_l2 > 0.0:
                l2_loss = tf.add_n(
                    [tf.nn.l2_loss(v) for v in train_variables if v.get_shape().ndims > 1])
                self.loss += config.lambda_l2 * l2_loss

            self.add_global_step = self.global_step.assign_add(config.batch_size)

            # Adam Optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate)
            grads_and_vars = tf.gradients(self.loss, train_variables)
            grads, _ = tf.clip_by_global_norm(
                t_list=grads_and_vars, clip_norm=self.grad_clip)
            self.train_op = optimizer.apply_gradients(zip(grads, train_variables))

    def _create_summaries(self):
        with tf.name_scope("natural_language_inference"):
            tf.summary.scalar("loss", self.loss)
            tf.summary.histogram("histogram loss", self.loss)
            self.summary_op = tf.summary.merge_all()

    def _create_log(self):
        log_file = './log/nli.log'
        handler = logging.FileHandler(log_file, mode='w')
        fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s'
        formatter = logging.Formatter(fmt)
        handler.setFormatter(formatter)
        self.logger = logging.getLogger('nli_logger')
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)

    def build_graph(self):
        # self._create_placeholder()
        self._get_data()
        self._create_embedding()
        self._create_forward()
        self._create_prediction()
        self._create_loss()
        self._create_optimizer()
        self._create_summaries()
        self._create_log()

    def train(self):
        # limit the usage of gpu
        # gpu_op = tf.GPUOptions(allow_growth=True)
        gpu_op = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        gpu_config = tf.ConfigProto(
            allow_soft_placement=False,
            log_device_placement=False,
            gpu_options=gpu_op)

        if self.is_continue:
            ckpt = tf.train.get_checkpoint_state(self.MODEL_FILE)
            if ckpt and ckpt.model_checkpoint_path:
                print("found model,continue training")
            else:
                print("model not found, please check your saved model")
                sys.exit()

        # lock the graph for read only
        #graph = tf.get_default_graph()
        #tf.Graph.finalize(graph)

        with tf.Session(config=gpu_config) as sess:
            if self.is_continue:
                saver = tf.train.Saver()
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("\ncontinue training model in mode")
            else:
                saver = tf.train.Saver(max_to_keep=1)
                sess.run(tf.global_variables_initializer())
                print("start training model in mode")
            writer = tf.summary.FileWriter('./graphs', sess.graph)

            for epoch_idx in range(config.epoch_size):
                batch_step = 0
                total_loss = 0.0
                total_correct = 0
                valid_preds = []
                valid_labels = []

                # train process
                sess.run(self.train_init)
                while True:
                    try:
                        self.global_step = sess.run(self.add_global_step)

                        loss_batch, num_correct, _, summary = sess.run(
                            [self.loss, self.num_correct,
                             self.train_op, self.summary_op])

                        total_loss += loss_batch
                        total_correct += num_correct
                        writer.add_summary(summary, global_step=self.global_step)

                        if (batch_step+1) % 99 == 0:
                            self.logger.debug(
                                'average_loss at epoch %d batch [%d:%d) : %3.5f ' \
                                '[acc: %3.4f]',
                                epoch_idx, batch_step-98, batch_step+1,
                                total_loss / 99,
                                total_correct / (config.batch_size*99))
                            print('average_loss at epoch %d batch [%d:%d) : ' \
                                  '%3.5f [acc: %3.4f]'
                                  %(epoch_idx, batch_step-98, batch_step+1,
                                    total_loss / 99,
                                    total_correct / (config.batch_size*99)))

                            total_loss = 0.0
                            total_correct = 0
                        batch_step += 1

                    except tf.errors.OutOfRangeError:
                        break

                # valid process
                sess.run(self.valid_init)
                while True:
                    try:
                        batch_preds, batch_labels = sess.run(
                            [self.predictions, self.labels])
                        valid_preds.extend(batch_preds)
                        valid_labels.extend(batch_labels)

                    except tf.errors.OutOfRangeError:
                        break

                # caculating precision, recall, F1, accuracy of valid data
                precision, recall, F1, accuracy = evaluation_rate(
                    valid_preds, valid_labels)
                self.logger.debug(
                    'evaluation at epoch %d : [precision: %3.4f], ' \
                    '[recall: %3.4f], [F1: %3.4f], [accuracy: %3.4f]',
                    epoch_idx, precision, recall, F1, accuracy)
                print('evaluation at epoch %d : [precision: %3.4f], '\
                      '[recall: %3.4f], [F1: %3.4f], [accuracy: %3.4f]'
                      %(epoch_idx, precision, recall, F1, accuracy))

                # save model per epoch
                saver.save(sess=sess,
                           save_path=self.MODEL_FILE + 'model.ckpt',
                           global_step=self.global_step,
                           write_meta_graph=True)
                self.logger.debug(
                    "model trained and saved at epoch %d \n", epoch_idx)
                print("model trained and saved at epoch %d \n" % epoch_idx)
"""
    def test(self, is_predict=True):
        # limit the usage of gpu
        # gpu_op = tf.GPUOptions(allow_growth=True)
        gpu_op = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        config = tf.ConfigProto(
            allow_soft_placement=False,
            log_device_placement=False,
            gpu_options=gpu_op)

        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(self.MODEL_FILE)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("\nthe model has been successfully restored")

                sess.run(self.test_init)
                test_preds = []
                test_labels = []
                while True:
                    try:
                        preds_batch, labels_batch = sess.run(
                            [self.predictions, self.labels])
                        test_preds.extend(preds_batch)
                        test_labels.extend(labels_batch)
                    except tf.errors.OutOfRangeError:
                        break

                if is_predict:
                    # test data without labels
                    with open("./result/result.csv", "w") as result_file:
                        print("test_id,result", file=result_file)
                        for idx, item in enumerate(test_preds):
                            print(str(idx)+','+str(item), file=result_file)

                    print("model test for prediction complete")
                    print("The percentage of label 1 is %3.4f" %
                          (float(sum(test_preds)/len(test_preds))))
                else:
                    # test data with labels
                    precision, recall, F1, accuracy = evaluation_rate(
                        test_preds, test_labels)
                    print('Extra evaluation: [precision: %3.4f], '\
                          '[recall: %3.4f], [F1: %3.4f], [accuracy: %3.4f]'
                          %(precision, recall, F1, accuracy))
            else:
                print("model restored failed")
"""
