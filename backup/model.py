# coding=utf-8

import sys
import tensorflow as tf
import logging.config
from nn import highway
from nn import self_attention
from nn import mlp
from utils import read_dataset
from utils import evaluation_rate


class Model:
    def __init__(self, batch_size, word_vocab_size, word_embed_size,
                 pos_vocab_size, pos_embed_size, max_len,
                 num_class, embedding_matrix_init, rnn_cell_size,
                 rnn_layers, att_da, att_r, mlp_hidden_size,learning_rate,
                 lambda_l2, grad_clip, is_continue=False):

        self.word_vocab_size = word_vocab_size
        self.pos_vocab_size = pos_vocab_size
        self.word_embed_size = word_embed_size
        self.pos_embed_size = pos_embed_size
        self.batch_size = batch_size
        self.max_len = max_len
        self.num_class = num_class
        self.embedding_matrix_init = embedding_matrix_init

        self.rnn_cell_size = rnn_cell_size
        self.rnn_layers = rnn_layers
        self.att_da = att_da
        self.att_r = att_r
        self.mlp_hidden_size = mlp_hidden_size
        self.lambda_l2 = lambda_l2
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.is_continue = is_continue

        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.global_step = tf.Variable(
            0, dtype=tf.int32, trainable=False, name='global_step')
        self.global_epoch = tf.Variable(
            0, dtype=tf.int32, trainable=False, name='global_epoch')
        self.MODEL_FILE = './save_model/'

    def _get_data(self):

        # read train_dataset
        train_folder = './data/ccks/train/'
        #train_data, valid_data = read_dataset(
        train_data, valid_data = read_dataset(
            passage_filename=train_folder+'passage.pkl',
            passage_pos_filename=train_folder+'passage_pos.pkl',
            passage_len_filename=train_folder+'passage_length.pkl',
            query_filename=train_folder+'query.pkl',
            query_pos_filename=train_folder+'query_pos.pkl',
            query_len_filename=train_folder+'query_length.pkl',
            label_filename=train_folder+'label.pkl',
            batch_size=self.batch_size)
        
        # read test_dataset
        test_folder = './data/ccks/test/'
        #test_folder = './data/ccks/extra_eval/'
        test_data = read_dataset(
            passage_filename=test_folder+'passage.pkl',
            passage_pos_filename=test_folder+'passage_pos.pkl',
            passage_len_filename=test_folder+'passage_length.pkl',
            query_filename=test_folder+'query.pkl',
            query_pos_filename=test_folder+'query_pos.pkl',
            query_len_filename=test_folder+'query_length.pkl',
            label_filename=test_folder+'label.pkl',
            batch_size=self.batch_size,
            is_train=False)

        iterator = tf.data.Iterator.from_structure(
            train_data.output_types, train_data.output_shapes)

        self.passage, self.passage_pos, self.passage_length,\
            self.query, self.query_pos, self.query_length,\
            self.labels = iterator.get_next()

        self.train_init = iterator.make_initializer(train_data)
        self.test_init = iterator.make_initializer(test_data)
        self.valid_init = iterator.make_initializer(valid_data)

        print("_get_data function worked")

    def _create_embedding(self):
        with tf.variable_scope('embedding', reuse=tf.AUTO_REUSE):
            # init embedding_matrix
            self.word_embeddings = tf.get_variable(
                name='word_embedding',
                shape=[self.word_vocab_size, self.word_embed_size],
                dtype=tf.float32,
                initializer=tf.random_uniform_initializer(-1.0, 1.0, seed=113),
                #initializer=tf.constant_initializer(self.embedding_matrix_init),
                trainable=True)

            self.pos_embeddings = tf.get_variable(
                name='pos_embedding',
                shape=[self.pos_vocab_size, self.pos_embed_size],
                dtype=tf.float32,
                initializer=tf.random_uniform_initializer(-1.0, 1.0, seed=113),
                trainable=True)

            #print("shape of passage == ", self.passage.shape.as_list())
            #print("shape of query == ", self.query.shape.as_list())

            # embedding_lookup
            self.passage_embed = tf.nn.dropout(
                tf.nn.embedding_lookup(self.word_embeddings, self.passage),
                self.keep_prob)
            self.query_embed = tf.nn.dropout(
                tf.nn.embedding_lookup(self.word_embeddings, self.query),
                self.keep_prob)
            """
            self.passage_pos_embed = tf.nn.dropout(
                tf.nn.embedding_lookup(self.pos_embeddings, self.passage_pos),
                self.keep_prob)
            self.query_pos_embed = tf.nn.dropout(
                tf.nn.embedding_lookup(self.pos_embeddings, self.query_pos),
                self.keep_prob)

            # concat word_embedding and pos_embedding
            # TODO test concat result
            self.passage_embed = tf.concat(
                [self.passage_word_embed, self.passage_pos_embed], axis=-1)
            self.query_embed = tf.concat(
                [self.query_word_embed, self.query_pos_embed], axis=-1)
            """
            # highway network layer
            self.passage_embed = highway(
                self.passage_embed,
                #size = self.rnn_hidden_units,
                scope="highway",
                keep_prob=self.keep_prob,
                reuse=None)
            self.query_embed = highway(
                self.query_embed,
                #size = self.rnn_hidden_units,
                scope="highway",
                keep_prob=self.keep_prob,
                reuse=True)

        print("_create_embedding and highway layer worked")

    def _create_bgrucell(self):
        with tf.variable_scope("bgru_layer"):
            cell_fw = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(
                num_units=self.rnn_cell_size,
                #kernel_initializer=tf.orthogonal_initializer(seed=115))
                kernel_initializer=tf.truncated_normal_initializer(
                    mean=0.0, stddev=0.1, seed=114))
            cell_bw = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(
                num_units=self.rnn_cell_size,
                #kernel_initializer=tf.orthogonal_initializer(seed=116))
                kernel_initializer=tf.truncated_normal_initializer(
                    mean=0.0, stddev=0.1, seed=114))

        print("_create_bgrucell function worked")
        return cell_fw, cell_bw

    def _create_blstm_cell(self):
        cell_fw = tf.contrib.rnn.LSTMCell(self.rnn_cell_size)
        cell_bw = tf.contrib.rnn.LSTMCell(self.rnn_cell_size)

        cell_fw = tf.contrib.rnn.DropoutWrapper(
            cell_fw, output_keep_prob=self.keep_prob)
        cell_bw = tf.contrib.rnn.DropoutWrapper(
            cell_bw, output_keep_prob=self.keep_prob)

        print("_create_blstm_cell function worked")
        return cell_fw, cell_bw

    def _create_forward(self):
        # single layer bgru encoder
        with tf.variable_scope('bigru', reuse=tf.AUTO_REUSE):
            # cell_fw, cell_bw = self._create_bgrucell()
            cell_fw, cell_bw = self._create_blstm_cell()
            self.passage_states, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.passage_embed,
                dtype=tf.float32,
                sequence_length=self.passage_length)
            self.query_states, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.query_embed,
                dtype=tf.float32,
                sequence_length=self.query_length)
            """
            self.ps_last = tf.concat(
                (self.passage_states[0][:, -1, :], self.passage_states[1][:, -1, :]),
                axis=-1)
            self.qs_last = tf.concat(
                (self.query_states[0][:, -1, :], self.query_states[0][:, -1, :]),
                axis=-1)
            """
            self.passage_states = tf.concat(self.passage_states, axis=-1)
            self.query_states = tf.concat(self.query_states, axis=-1)
            # print("shape of self.passage_states", self.passage_states.shape.as_list())

        print("bgru encoder layer worked")

        # self-attention
        with tf.variable_scope('self_ATT', reuse=tf.AUTO_REUSE):
            self.passage_vec, _ = self_attention(
                inputs=self.passage_states,
                da=self.att_da, r=self.att_r,
                batch_size=self.batch_size,
                reuse=tf.AUTO_REUSE)

            self.query_vec, _ = self_attention(
                inputs=self.query_states,
                da=self.att_da, r=self.att_r,
                batch_size=self.batch_size,
                reuse=tf.AUTO_REUSE)

        print("self-attention layer worked")

        # Multi Layer Perceptron
        with tf.variable_scope('MLP', reuse=tf.AUTO_REUSE):
            self.mlp_inputs = tf.concat(
                [self.passage_vec, self.query_vec], axis=-1)
            #print("self.p_fs.shape == ", self.ps_last.shape.as_list())
            #self.mlp_inputs = tf.concat(
            #    (self.ps_last, self.qs_last), axis=-1)
            self.logits = mlp(
                mlp_inputs=self.mlp_inputs,
                mlp_hidden_size=self.mlp_hidden_size,
                num_class=self.num_class,
                keep_prob=self.keep_prob)
        print("MLP layer worked")

    def _create_prediction(self):
        with tf.name_scope("prediction"):
            prob = tf.nn.softmax(self.logits)
            self.predictions = tf.argmax(prob, 1)
            #print("shape of self.predictions == ", self.predictions.shape.as_list())
            #print("shape of self.labels == ", self.labels.shape.as_list())
            #correct = tf.nn.in_top_k(self.logits, self.labels, 1)
            correct = tf.equal(self.predictions, self.labels)
            self.num_correct = tf.reduce_sum(tf.cast(correct, tf.int32))

    def _create_loss(self):
        with tf.name_scope("loss"):
            gold_matrix = tf.one_hot(self.labels, self.num_class, dtype=tf.float32)
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.logits, labels=gold_matrix)
            )

    def _create_optimizer(self):
        with tf.name_scope("optimizer"):
            train_variables = tf.trainable_variables()

            # L2 norm
            if self.lambda_l2 > 0.0:
                l2_loss = tf.add_n(
                    [tf.nn.l2_loss(v) for v in train_variables if v.get_shape().ndims > 1])
                self.loss += self.lambda_l2 * l2_loss

            self.add_global_step = self.global_step.assign_add(self.batch_size)

            # Adam Optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            # grads_and_vars = optimizer.compute_gradients(self.loss, train_variables)
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

    def train(self, epoch_total, keep_prob):
        # limit the usage of gpu
        # gpu_op = tf.GPUOptions(allow_growth=True)
        gpu_op = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        config = tf.ConfigProto(allow_soft_placement=False,
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

        with tf.Session(config=config) as sess:
            if self.is_continue:
                saver = tf.train.Saver()
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("\ncontinue training model in mode")
            else:
                saver = tf.train.Saver(max_to_keep=1)
                sess.run(tf.global_variables_initializer())
                print("start training model in mode")
            writer = tf.summary.FileWriter('./graphs', sess.graph)

            for epoch_idx in range(epoch_total):
                batch_step = 0
                total_loss = 0.0
                total_correct = 0
                valid_preds = []
                valid_labels = []
                train_feed_dict = {self.keep_prob: keep_prob}
                valid_feed_dict = {self.keep_prob: 1.0}

                # train process
                sess.run(self.train_init)
                while True:
                    try:
                        self.global_step = sess.run(self.add_global_step)

                        loss_batch, num_correct, _, summary = sess.run(
                            [self.loss, self.num_correct, self.train_op, self.summary_op],
                            feed_dict=train_feed_dict)

                        total_loss += loss_batch
                        total_correct += num_correct
                        writer.add_summary(summary, global_step=self.global_step)

                        if (batch_step+1) % 20 == 0:
                            self.logger.debug(
                                'average_loss at epoch %d batch [%d:%d) : %3.5f',
                                epoch_idx, batch_step-19, batch_step+1,
                                total_loss / 20)
                            print('average_loss at epoch %d batch [%d:%d) : %3.5f [acc: %3.4f]'
                                  %(epoch_idx, batch_step-19, batch_step+1,
                                    total_loss / 20,
                                    total_correct / (self.batch_size*20)))

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
                            [self.predictions, self.labels], feed_dict=valid_feed_dict)
                        valid_preds.extend(batch_preds)
                        valid_labels.extend(batch_labels)

                    except tf.errors.OutOfRangeError:
                        break

                # caculating precision, recall, F1, accuracy of valid data
                precision, recall, F1, accuracy = evaluation_rate(
                    valid_preds, valid_labels)
                self.logger.debug(
                    'evaluation at epoch %d : [precision: %3.4f], [recall: %3.4f], '\
                    '[F1: %3.4f], [accuracy: %3.4f]',
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
                test_feed_dict = {self.keep_prob: 1.0}
                while True:
                    try:
                        preds_batch, labels_batch = sess.run(
                            [self.predictions, self.labels], feed_dict=test_feed_dict)
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
