# coding=utf-8

import tensorflow as tf
from nn import mlp

class BiLSTM_Model:
    def __init__(self, vocab_size, embed_size, batch_size, max_len,
                 num_class, embedding_matrix_init, rnn_cell_size,
                 rnn_layers, mlp_hidden_size, learning_rate, lambda_l2):

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.max_len = max_len
        self.num_class = num_class
        self.embedding_matrix_init = embedding_matrix_init

        self.rnn_cell_size = rnn_cell_size
        self.rnn_layers = rnn_layers
        self.mlp_hidden_size = mlp_hidden_size
        self.lambda_l2 = lambda_l2
        self.learning_rate = learning_rate

        self.global_step = tf.Variable(
            0, dtype=tf.int32, trainable=False, name='global_step')

    def _create_placeholder(self):
        with tf.name_scope("data_of_bilstm_model"):
            self.passage = tf.placeholder(
                shape=(self.batch_size, self.max_len), dtype=tf.int32, name='passage')
            self.query = tf.placeholder(
                shape=(self.batch_size, self.max_len), dtype=tf.int32, name='query')
            self.label = tf.placeholder(
                shape=(self.batch_size,), dtype=tf.int32, name='label')
            self.passage_length = tf.placeholder(
                shape=(self.batch_size,), dtype=tf.int32, name='passage_length')
            self.query_length = tf.placeholder(
                shape=(self.batch_size,), dtype=tf.int32, name='query_length')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # print("_create_placeholder function worked")

    def _create_embedding(self):
        with tf.variable_scope('embedding', reuse=tf.AUTO_REUSE):
            # init embedding_matrix
            self.embeddings = tf.get_variable(
                name='embedding_matrix',
                shape=[self.vocab_size, self.embed_size],
                dtype=tf.float32,
                initializer=tf.random_uniform_initializer(-1.0, 1.0, seed=113),
                #initializer=tf.constant_initializer(self.embedding_matrix_init),
                trainable=True)

            # embedding_lookup
            self.passage_embed = tf.nn.dropout(
                tf.nn.embedding_lookup(self.embeddings, self.passage),
                self.keep_prob)
            self.query_embed = tf.nn.dropout(
                tf.nn.embedding_lookup(self.embeddings, self.query),
                self.keep_prob)

        # print("_create_embedding function worked")

    def _create_blstm_cell(self):
        cell_fw = tf.contrib.rnn.LSTMCell(self.rnn_cell_size)
        cell_bw = tf.contrib.rnn.LSTMCell(self.rnn_cell_size)

        cell_fw = tf.contrib.rnn.DropoutWrapper(
            cell_fw, output_keep_prob=self.keep_prob)
        cell_bw = tf.contrib.rnn.DropoutWrapper(
            cell_bw, output_keep_prob=self.keep_prob)
        
        # print("_create_blstm_cell function worked")
        return cell_fw, cell_bw

    def _create_forward(self):
        #  bidirectional lstm encoder
        with tf.variable_scope('bi_lstm', reuse=tf.AUTO_REUSE):
            cell_fw, cell_bw = self._create_blstm_cell()
            
            passage_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.passage_embed,
                dtype=tf.float32,
                sequence_length=self.passage_length)

            query_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.query_embed,
                dtype=tf.float32,
                sequence_length=self.query_length)

            passage_outputs_fw, passage_outputs_bw = passage_outputs
            query_outputs_fw, query_outputs_bw = query_outputs

            self.passage_outputs_last = tf.concat(
                (passage_outputs_fw[:, -1, :], passage_outputs_bw[:, -1, :]),
                axis=-1)
            self.query_outputs_last = tf.concat(
                (query_outputs_fw[:, -1, :], query_outputs_bw[:, -1, :]),
                axis=-1)

            # print("shape of passage_outputs_last", passage_outputs_last.shape.as_list())
        #print(" bi-lstm layer worked")

        # Multi Layer Perceptron
        with tf.variable_scope('MLP', reuse=tf.AUTO_REUSE):
            self.mlp_inputs = tf.concat(
                (self.passage_outputs_last, self.query_outputs_last), axis=-1)
            self.logits = mlp(
                mlp_inputs=self.mlp_inputs,
                mlp_hidden_size=self.mlp_hidden_size,
                num_class=self.num_class,
                keep_prob=self.keep_prob)

        #print("MLP layer worked")

    def _create_prediction(self):
        with tf.name_scope("prediction"):
            prob = tf.nn.softmax(self.logits)
            self.prediction = tf.argmax(prob, axis=1, output_type=tf.int32)

        #print("_create_prediction function worked")

    def _create_loss(self):
        with tf.name_scope("loss"):
            gold_matrix = tf.one_hot(self.label, self.num_class, dtype=tf.float32)
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.logits, labels=gold_matrix))

        #print("_create_loss function worked")

    def _create_optimizer(self):
        with tf.name_scope("optimizer"):
            train_variable = tf.trainable_variables()

            # L2 norm
            if self.lambda_l2 > 0.0:
                l2_loss = tf.add_n(
                    [tf.nn.l2_loss(v) for v in train_variable if v.get_shape().ndims > 1])
                self.loss += self.lambda_l2 * l2_loss

            self.add_global_step = self.global_step.assign_add(self.batch_size)

            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate).minimize(self.loss)

        #print("_create_optimizer function worked")

    def build_graph(self):
        self._create_placeholder()
        self._create_embedding()
        self._create_forward()
        self._create_prediction()
        self._create_loss()
        self._create_optimizer()

        #print("build_graph function worked")

