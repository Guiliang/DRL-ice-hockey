import tensorflow as tf


class td_prediction_tt_embed:
    def __init__(self, feature_number, h_size, max_trace_length, learning_rate,
                 output_layer_size=3,
                 lstm_layer_num=2,
                 dense_layer_num=2,
                 model_name="tt_lstm",
                 rnn_type="bp_last_step"):
        """
        init the model
        """
        self.feature_number = feature_number
        self.h_size = h_size
        self.max_trace_length = max_trace_length
        self.learning_rate = learning_rate
        self.rnn_type = rnn_type
        self.model_name = model_name
        self.lstm_layer_num = lstm_layer_num
        self.dense_layer_num = dense_layer_num
        self.output_layer_size = output_layer_size

        self.rnn_input_ph = None
        self.trace_lengths_ph = None
        self.home_away_indicator_ph = None
        self.y_ph = None

        self.lstm_cell_home_all = []
        self.lstm_cell_away_all = []
        self.dense_layer_weights = []
        self.dense_layer_bias = []
        self.embed_away_b = None
        self.embed_away_w = None
        self.embed_home_w = None
        self.embed_home_b = None
        self.readout = None

    def build(self):
        """
        define a shallow dynamic LSTM
        """
        with tf.name_scope(self.model_name):
            with tf.name_scope("tower-for-home"):
                with tf.name_scope("Home-LSTM-layer"):
                    for i in range(self.lstm_layer_num):
                        self.lstm_cell_home_all.append(
                            tf.nn.rnn_cell.LSTMCell(num_units=self.h_size, state_is_tuple=True,
                                                    initializer=tf.random_normal_initializer()))
                with tf.name_scope("Home_embed_layer"):
                    self.embed_home_w = tf.get_variable('w_embed_home', [self.h_size, self.h_size],
                                                        initializer=tf.contrib.layers.xavier_initializer())
                    self.embed_home_b = tf.Variable(tf.zeros([self.h_size]), name="b_embed_home")

            with tf.name_scope("tower-for-away"):
                with tf.name_scope("Away-LSTM-layer"):
                    for i in range(self.lstm_layer_num):
                        self.lstm_cell_away_all.append(
                            tf.nn.rnn_cell.LSTMCell(num_units=self.h_size, state_is_tuple=True,
                                                    initializer=tf.random_normal_initializer()))
                with tf.name_scope("Away-embed-layer"):
                    self.embed_away_w = tf.get_variable('w_embed_away', [self.h_size, self.h_size],
                                                        initializer=tf.contrib.layers.xavier_initializer())
                    self.embed_away_b = tf.Variable(tf.zeros([self.h_size]), name="b_embed_away")

            with tf.name_scope("Dense_Layer"):
                for i in range(self.dense_layer_num):
                    w_input_size = self.h_size
                    w_output_size = self.h_size if i < self.dense_layer_num - 1 else self.output_layer_size
                    self.dense_layer_weights.append(tf.get_variable('w{0}_xaiver'.format(str(i)),
                                                                    [w_input_size, w_output_size],
                                                                    initializer=tf.contrib.layers.xavier_initializer()))
                    self.dense_layer_bias.append(tf.Variable(tf.zeros([w_output_size]), name="b_{0}".format(str(i))))

    def call(self):
        """
        build the network
        :return: 
        """
        with tf.name_scope(self.model_name):
            with tf.name_scope("tower-for-home"):
                with tf.name_scope("Home_LSTM_layer"):
                    rnn_output = None
                    for i in range(self.lstm_layer_num):
                        rnn_input = self.rnn_input_ph if i == 0 else rnn_output
                        rnn_output, rnn_state = tf.nn.dynamic_rnn(  # while loop dynamic learning rnn
                            inputs=rnn_input, cell=self.lstm_cell_home_all[i],
                            sequence_length=self.trace_lengths_ph, dtype=tf.float32,
                            scope=self.rnn_type + '_home_rnn_{0}'.format(str(i)))
                    outputs = tf.stack(rnn_output)
                    # Hack to build the indexing and retrieve the right output.
                    self.batch_size = tf.shape(outputs)[0]
                    # Start indices for each sample
                    self.index = tf.range(0, self.batch_size) * self.max_trace_length + (self.trace_lengths_ph - 1)
                    # Indexing
                    rnn_last = tf.gather(tf.reshape(outputs, [-1, self.h_size]), self.index)

                with tf.name_scope("Home_embed_layer"):
                    self.home_embed_layer = tf.matmul(rnn_last, self.embed_home_w) + self.embed_home_b

            with tf.name_scope("tower-for-away"):
                with tf.name_scope("Away_LSTM_layer"):
                    rnn_output = None
                    for i in range(self.lstm_layer_num):
                        rnn_input = self.rnn_input_ph if i == 0 else rnn_output
                        rnn_output, rnn_state = tf.nn.dynamic_rnn(  # while loop dynamic learning rnn
                            inputs=rnn_input, cell=self.lstm_cell_away_all[i],
                            sequence_length=self.trace_lengths_ph, dtype=tf.float32,
                            scope=self.rnn_type + '_away_rnn_{0}'.format(str(i)))
                    outputs = tf.stack(rnn_output)
                    # Hack to build the indexing and retrieve the right output.
                    self.batch_size = tf.shape(outputs)[0]
                    # Start indices for each sample
                    self.index = tf.range(0, self.batch_size) * self.max_trace_length + (self.trace_lengths_ph - 1)
                    # Indexing
                    rnn_last = tf.gather(tf.reshape(outputs, [-1, self.h_size]), self.index)

                with tf.name_scope("Away_embed_layer"):
                    self.away_embed_layer = tf.matmul(rnn_last, self.embed_away_w) + self.embed_away_b

            embed_layer = tf.where(condition=self.home_away_indicator_ph,
                                   x=self.home_embed_layer,
                                   y=self.away_embed_layer)

            with tf.name_scope('dense-layer'):
                dense_output = None
                for i in range(self.dense_layer_num):
                    dense_input = embed_layer if i == 0 else dense_output
                    dense_output = tf.matmul(dense_input, self.dense_layer_weights[i]) + self.dense_layer_bias[i]
                    dense_output = tf.nn.relu(dense_output, name='activation_{0}'.format(str(i)))

            self.readout = dense_output
            with tf.name_scope("cost"):
                self.cost = tf.reduce_mean(tf.square(self.y_ph - self.readout))
                self.diff = tf.reduce_mean(tf.abs(self.y_ph - self.readout))
            tf.summary.histogram('cost', self.cost)

            with tf.name_scope("train"):
                self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    def initialize_ph(self):
        """
        initialize the place holder
        :return: 
        """
        rnn_input_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.max_trace_length,
                                                               self.feature_number], name="rnn-input-ph")
        trace_lengths_ph = tf.placeholder(dtype=tf.int32, shape=[None], name="trace-length")
        home_away_indicator_ph = tf.cast(tf.placeholder(dtype=tf.int32, shape=[None], name="indicator-ph"), tf.bool)
        y_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.output_layer_size])

        self.rnn_input_ph = rnn_input_ph
        self.trace_lengths_ph = trace_lengths_ph
        self.home_away_indicator_ph = home_away_indicator_ph
        self.y_ph = y_ph


if __name__ == '__main__':
    """testing the model locally"""
    test_model = td_prediction_tt_embed(feature_number=25, h_size=32, max_trace_length=10, learning_rate=0.0001)
    test_model.initialize_ph()
    test_model.build()
    test_model.call()
    print "testing model"
