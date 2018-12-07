import tensorflow as tf
import tensorflow.contrib as tc

class BiLSTM(object):
    def __init__(self, sequence_length, num_classes, vocab_size,
                 embedding_dim, num_layers, hidden_dim, learning_rate):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # Embedding layer 指定在cpu
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            self.embedding_words = tf.Variable(tf.random_uniform([vocab_size, embedding_dim], -1.0, 1.0), name='embedding_words')
            self.embeded_chars = tf.nn.embedding_lookup(self.embedding_words, self.input_x)

        # bilstm
        #含有dropout的lstm cell
        def lstm_cell_dropout():
            cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim, state_is_tuple=True)
            return cell
            # cell = tf.contrib.rnn.BasicLSTMCell(hidden_dim, state_is_tuple=True)
            # return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob)

        with tf.name_scope('bi_lstm'):
            #建立前后多向层cells
            cells_fw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_dropout() for _ in range(num_layers)], state_is_tuple=True)
            cells_bw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_dropout() for _ in range(num_layers)], state_is_tuple=True)
            # cells_fw = tf.contrib.rnn.MultiRNNCell([lstm_cell_dropout() for _  in range(num_layers)], state_is_tuple=True)
            # cells_bw = tf.contrib.rnn.MultiRNNCell([lstm_cell_dropout() for _ in range(num_layers)], state_is_tuple=True)

            _outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cells_fw, cell_bw=cells_bw, inputs=self.embeded_chars, dtype=tf.float32)
            # 取最后一个时序作为结果
            last = _outputs[:, -1, :]

        with tf.name_scope('score'):
            #relu 激活层
            fc = tf.layers.dense(input=last, units=hidden_dim, name='fc1' )
            fc = tf.nn.dropout(fc, keep_prob=self.dropout_keep_prob)
            fc =tf.nn.relu(fc)

            #classifier
            self.logits = tf.layers.dense(input=fc, units=num_classes, name='fc2')
            #probability
            self.prob = tf.nn.softmax(self.logits)
            #prediction softmax的结果是m*num_classes 所以取X方向的最大值，即某一样本的中的最大值
            self.y_pred = tf.arg_max(self.prob,1)

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits + 1e-10, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            #optimizer
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.y_pred, self.input_y)
            self.accuracy = tf.reduce_mean(tf.case(correct_predictions, tf.float32))






