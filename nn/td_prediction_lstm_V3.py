import tensorflow as tf


class td_prediction_lstm_V3:
    def __init__(self, FEATURE_NUMBER, H_SIZE, MAX_TRACE_LENGTH, learning_rate, rnn_type='bp_last_step'):
        """
        define a shallow dynamic LSTM
        """
        with tf.name_scope("LSTM_layer"):
            self.rnn_input = tf.placeholder(tf.float32, [None, 10, FEATURE_NUMBER], name="x_1")
            self.trace_lengths = tf.placeholder(tf.int32, [None], name="tl")

            self.lstm_cell = tf.contrib.rnn.LSTMCell(num_units=H_SIZE, state_is_tuple=True,
                                                     initializer=tf.random_uniform_initializer(-0.05, 0.05))

            self.rnn_output, self.rnn_state = tf.nn.dynamic_rnn(  # while loop dynamic learning rnn
                inputs=self.rnn_input, cell=self.lstm_cell, sequence_length=self.trace_lengths, dtype=tf.float32,
                scope=rnn_type + '_rnn')

            # [batch_size, max_time, cell.output_size]
            self.outputs = tf.stack(self.rnn_output)

            # Hack to build the indexing and retrieve the right output.
            self.batch_size = tf.shape(self.outputs)[0]
            # Start indices for each sample
            self.index = tf.range(0, self.batch_size) * MAX_TRACE_LENGTH + (self.trace_lengths - 1)
            # Indexing
            self.rnn_last = tf.gather(tf.reshape(self.outputs, [-1, H_SIZE]), self.index)

        num_layer_1 = H_SIZE
        num_layer_2 = 3

        with tf.name_scope("Dense_Layer_first"):
            self.W1 = tf.get_variable('w1_xaiver', [num_layer_1, num_layer_2],
                                      initializer=tf.contrib.layers.xavier_initializer())
            self.b1 = tf.Variable(tf.zeros([num_layer_2]), name="b_1")
            self.read_out = tf.matmul(self.rnn_last, self.W1) + self.b1
            # self.activation1 = tf.nn.relu(self.y1, name='activation')

        self.y = tf.placeholder("float", [None, num_layer_2])

        with tf.name_scope("cost"):
            self.readout_action = self.read_out
            self.cost = tf.reduce_mean(tf.square(self.y - self.readout_action))
            self.diff = tf.reduce_mean(tf.abs(self.y - self.readout_action))
        tf.summary.histogram('cost', self.cost)

        with tf.name_scope("train"):
            self.train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
