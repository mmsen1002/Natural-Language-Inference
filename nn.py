import tensorflow as tf


initializer = lambda: tf.contrib.layers.variance_scaling_initializer(
    factor=1.0,
    mode='FAN_AVG',
    uniform=True,
    dtype=tf.float32)
initializer_relu = lambda: tf.contrib.layers.variance_scaling_initializer(
    factor=2.0,
    mode='FAN_IN',
    uniform=False,
    dtype=tf.float32)
regularizer = tf.contrib.layers.l2_regularizer(scale=3e-7)


def conv(inputs, output_size, bias=None,
         activation=None, kernel_size=1,
         name="conv", reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        shapes = inputs.shape.as_list()
        if len(shapes) > 4:
            raise NotImplementedError
        elif len(shapes) == 4:
            filter_shape = [1, kernel_size, shapes[-1], output_size]
            bias_shape = [1, 1, 1, output_size]
            strides = [1, 1, 1, 1]
        else:
            filter_shape = [kernel_size, shapes[-1], output_size]
            bias_shape = [1, 1, output_size]
            strides = 1
        conv_func = tf.nn.conv1d if len(shapes) == 3 else tf.nn.conv2d
        kernel_ = tf.get_variable(
            "kernel_",
            filter_shape,
            dtype=tf.float32,
            regularizer=regularizer,
            initializer=initializer_relu() if activation is not None else initializer())
        outputs = conv_func(inputs, kernel_, strides, "VALID")
        if bias:
            outputs += tf.get_variable(
                "bias_",
                bias_shape,
                regularizer=regularizer,
                initializer=tf.zeros_initializer())
        if activation is not None:
            return activation(outputs)
        else:
            return outputs


def highway(x, size=None, activation=tf.nn.relu, num_layers=2,
            scope="highway", keep_prob=1.0,  is_train=True, reuse=None):
    with tf.variable_scope(scope, reuse):
        if size is None:
            size = x.shape.as_list()[-1]
        else:
            x = conv(x, size, name="input_projection", reuse=reuse)
        for i in range(num_layers):
            T = conv(x, size, bias=True, activation=tf.sigmoid,
                     name="gate_%d"%i, reuse=reuse)
            H = conv(x, size, bias=True, activation=activation,
                     name="activation_%d"%i, reuse=reuse)
            if is_train:
                H = tf.nn.dropout(H, keep_prob)
            x = H * T + x * (1.0 - T)
        return x


def mask_op(inputs, seq_len, max_seq_len):
    bottom = tf.ones_like(inputs) * (-2**32+1)
    mask = tf.sequence_mask(seq_len, max_seq_len)
    mask = tf.tile(mask, [1, max_seq_len])
    mask = tf.stack(tf.split(mask, max_seq_len, axis=1))
    mask = tf.transpose(mask, perm=[1, 2, 0])
    masked = tf.where(tf.equal(mask, True), inputs, bottom)
    return masked


def self_attention(inputs, da, r, batch_size):

    # shape(inputs) == [batch_size, seq_len, per_vec_size]
    _, seq_len, per_vec_size = inputs.shape.as_list()
    # print("shape of inputs == ", inputs.shape.as_list())

    # TODO replace initializer
    att_initializer = tf.random_normal_initializer(
        mean=0.0, stddev=1.0, seed=116, dtype=tf.float32)
    # initializer = tf.random_uniform_initializer(-1.0, 1.0, seed=113)

    Ws1 = tf.get_variable(
        name="Ws1",
        shape=[per_vec_size, da],
        initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, seed=101))
    Ws2 = tf.get_variable(
        name="Ws2",
        shape=[da, r],
        initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, seed=102))
    # shape(inputs_reshaped) == [batch_size*seq_len, per_vec_size]
    inputs_reshaped = tf.reshape(
        inputs,
        shape=[-1, per_vec_size],
        name='hidden_units_reshaped')
    # shape(inputs_transposed) == [per_vec_size, batch_size*seq_len]
    # inputs_transposed = tf.transpose(inputs_reshaped, [1, 0])
    # shape(tanh_Ws1_time_X) == [batch_size*seq_len, da]
    tanh_Ws1_time_X = tf.nn.tanh(
        tf.matmul(inputs_reshaped, Ws1), name="tanh_Ws1_time_X")
    # shape(tanh_Ws1_time_X_and_time_Ws2) == [batch_size*seq_len, r]
    tanh_Ws1_time_X_and_time_Ws2 = tf.matmul(
        tanh_Ws1_time_X, Ws2,  name="tanh_ws1_time_X_and_time_Ws2")
    # shape(att) == [batch_size, r, seq_len]
    att = tf.nn.softmax(
        tf.reshape(
            tanh_Ws1_time_X_and_time_Ws2,
            shape=[batch_size, r, seq_len],
            name="reshape_tanh_Ws1_time_X_and_time_Ws2"),
        #dim=1,
        name="self_attention")
    # shape(output_matrix) = [batch_size, r, per_vec_size]
    output_matrix = tf.matmul(att, inputs)
    print(output_matrix.shape.as_list())
    # for r==1, transfer output_matrix to output_vec
    # shape(output_vec) == [batch_size, per_vec_size]
    #output_vec = tf.reshape(output_matrix, shape=[-1, per_vec_size])

    return output_matrix, att


def bidaf(passage, query, batch_size):

    _, seq_len, context_dim = passage.shape.as_list()
    W = tf.truncated_normal(
        shape=[seq_len, seq_len], mean=0.0, stddev=1.0,
        dtype=tf.float32, seed=15, name="weight")

    # mul.shape == [batch_size, seq_len, seq_len]
    mul = tf.matmul(passage, query, transpose_b=True)

    # passage to query
    # p2q_mul.shape = [batch_size*seq_len, seq_len]
    p2q_mul = tf.matmul(tf.reshape(mul, shape=[-1, seq_len]), W)
    # p2q_att.shape = [batch_size, seq_len, seq_len]
    p2q_att = tf.nn.softmax(
        tf.reshape(p2q_mul, shape=[batch_size, seq_len, seq_len]),
        dim=-1,
        name="passage2query_attention")
    # passage2query.shape == [batch_size, seq_len, context_dim]
    passage2query = tf.matmul(p2q_att, query)
    passage2query = tf.concat(
        [passage, passage2query], -1)

    # query to passage
    # q2p_att.shape = [batch_size*seq_len, seq_len]
    q2p_mul = tf.matmul(
        tf.reshape(tf.transpose(mul, perm=[0, 2, 1]), shape=[-1, seq_len]), W)
    # q2p_att.shape = [batch_size, seq_len, seq_len]
    q2p_att = tf.nn.softmax(
        tf.reshape(q2p_mul, shape=[batch_size, seq_len, seq_len]),
        dim=-1,
        name="query2passage_attention")
    # query2passage.shape = [batch_size, seq_len, context_dim]
    query2passage = tf.matmul(q2p_att, passage)
    query2passage = tf.concat(
        [query, query2passage], -1)

    # concat outputs
    # concat_outputs.shape = [batch_size, seq_len, context_dim*2]
    #concat_outputs = tf.concat(
    #    [passage2query, query2passage], -1)

    return passage2query, query2passage


def match(passage_encodes, query_encodes):
    """
    Match the passage_encodes with query_encodes using BiDAF algorithm
    """
    with tf.variable_scope('bidaf'):
        sim_matrix = tf.matmul(passage_encodes, query_encodes, transpose_b=True)
        passage2query_att = tf.matmul(tf.nn.softmax(sim_matrix, -1), query_encodes)
        b = tf.nn.softmax(tf.expand_dims(tf.reduce_max(sim_matrix, 2), 1), -1)
        query2passage_att = tf.tile(tf.matmul(b, passage_encodes),
                                    [1, tf.shape(passage_encodes)[1], 1])
        concat_outputs = tf.concat([passage2query_att, query2passage_att], -1)
        return concat_outputs


def esim_match(p_encoded, q_encoded, p_len, q_len,
               max_seq_len, keep_prob, rnn_cell_size,
               is_train):
    """
    match the p_encoded with q_encoded using ESIM algorithm
    """
    with tf.variable_scope('matching'):
        att_matrix = tf.matmul(p_encoded, q_encoded, transpose_b=True)
        p_att = tf.nn.softmax(mask_op(att_matrix, q_len, max_seq_len), dim=1)
        p_align = tf.matmul(p_att, q_encoded)

        att_matrix = tf.transpose(att_matrix, perm=[0, 2, 1])
        q_att = tf.nn.softmax(mask_op(att_matrix, p_len, max_seq_len), dim=1)
        q_align = tf.matmul(q_att, p_encoded)

        p_combined = tf.concat(
            [p_encoded, p_encoded-p_align, p_encoded*p_align], axis=2)
        q_combined = tf.concat(
            [q_encoded, q_encoded-q_align, q_encoded*q_align], axis=2)

        if is_train:
            p_combined = tf.nn.dropout(
                p_combined, keep_prob=keep_prob)
            q_combined = tf.nn.dropout(
                q_combined, keep_prob=keep_prob)

    with tf.variable_scope('aggregation'):
        fw_cell_comp = tf.nn.rnn_cell.GRUCell(rnn_cell_size)
        bw_cell_comp = tf.nn.rnn_cell.GRUCell(rnn_cell_size)

        p_outputs_comp, _ = tf.nn.bidirectional_dynamic_rnn(
            fw_cell_comp,
            bw_cell_comp,
            p_combined,
            sequence_length=p_len,
            dtype=tf.float32,
            scope='aggregation')
        q_outputs_comp, _ = tf.nn.bidirectional_dynamic_rnn(
            fw_cell_comp,
            bw_cell_comp,
            q_combined,
            sequence_length=q_len,
            dtype=tf.float32,
            scope='aggregation')

        p_outputs_comp = tf.concat(p_outputs_comp, axis=2)
        q_outputs_comp = tf.concat(q_outputs_comp, axis=2)

    with tf.variable_scope('pooling'):
        p_ave = tf.reduce_sum(p_outputs_comp, axis=1)
        p_ave = p_ave / (tf.cast(tf.expand_dims(p_len, axis=1), tf.float32))

        # q_ave = tf.reduce_mean(q_outputs_comp, axis=2)
        q_ave = tf.reduce_sum(q_outputs_comp, axis=1)
        q_ave = q_ave /(tf.cast(tf.expand_dims(q_len, axis=1), tf.float32))

        p_max = tf.reduce_max(p_outputs_comp, axis=1)
        q_max = tf.reduce_max(q_outputs_comp, axis=1)
        pooling_flat = tf.concat([p_ave, q_ave, p_max, q_max], axis=1)
        #pooling_flat = tf.layers.batch_normalization(pooling_flat)

    return pooling_flat


def mlp(mlp_inputs, mlp_hidden_size, num_class, keep_prob, is_train):

    # shape(mlp_inputs) == [batch_size, seq_len, context_dim]
    batch_size, vector_dim = mlp_inputs.shape.as_list()
    #mlp_inputs = tf.reshape(mlp_inputs, shape=[-1, context_dim])

    # TODO replace initializer
    #mlp_initializer = tf.random_normal_initializer(
    #    mean=0.0, stddev=1.0, seed=115, dtype=tf.float32)
    weight = {
        'h1': tf.get_variable(
            name="weight_h1",
            shape=[vector_dim, mlp_hidden_size],
            initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, seed=117)),
        'h2': tf.get_variable(
            name="weight_h2",
            shape=[mlp_hidden_size, mlp_hidden_size],
            initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, seed=118)),
        'out': tf.get_variable(
            name="weight_out",
            shape=[mlp_hidden_size, num_class],
            initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, seed=119))
    }
    bias = {
        'h1': tf.get_variable(
            name="bias_h1",
            shape=[mlp_hidden_size],
            initializer=tf.zeros_initializer()),
        'h2': tf.get_variable(
            name="bias_h2",
            shape=[mlp_hidden_size],
            initializer=tf.zeros_initializer()),
        'out': tf.get_variable(
            name="bias_out",
            shape=[num_class],
            initializer=tf.zeros_initializer())
        }
    if is_train:
        mlp_inputs = tf.nn.dropout(mlp_inputs, keep_prob)

    layer1 = tf.nn.relu(
        tf.add(tf.matmul(mlp_inputs, weight['h1']), bias['h1']))
    if is_train:
        layer1 = tf.nn.dropout(layer1, keep_prob)
    """
    layer2 = tf.nn.relu(
        tf.add(tf.matmul(layer1, weight['h2']), bias['h2']))
    layer2 = tf.nn.dropout(layer2, keep_prob)
    """
    # shape(output_layer) == [batch_size, num_lass]
    output_layer = tf.add(tf.matmul(layer1, weight['out']), bias['out'])

    return output_layer


def collect_final_step_of_lstm(lstm_representation, lengths):
    # lstm_representation: [batch_size, passsage_length, dim]
    # lengths: [batch_size]
    lengths = tf.cast(lengths, tf.int32)
    lengths = tf.maximum(lengths, tf.zeros_like(lengths, dtype=tf.int32))

    batch_size = tf.shape(lengths)[0]
    batch_nums = tf.range(0, limit=batch_size) # shape (batch_size)
    indices = tf.stack((batch_nums, lengths), axis=1) # shape (batch_size, 2)
    result = tf.gather_nd(lstm_representation, indices, name='last-forwar-lstm')

    return result # [batch_size, dim]


if __name__ == '__main__':
    input_seqs = tf.random_normal(
        shape=[4, 7, 5], dtype=tf.float32, stddev=1, seed=1)
    #outputs_hw = highway(input_seqs)
    outputs_vec, att = self_attention(
        input_seqs, 13, 1, 4)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(outputs_vec))
