#! /user/bin/evn python
# -*- coding:utf8 -*-

"""

@Author : Lau James
@Contact : LauJames2017@whu.edu.cn
@Project : Structure_Func_Recognition 
@File : trainTextRNN.py
@Time : 1/20/18 2:36 PM
@Software : PyCharm
@Copyright: "Copyright (c) 2017 Lau James. All Rights Reserved"
"""

import tensorflow as tf
import numpy as np
import os
import dataHelper
import time
import datetime
from textRNN import TextRNN

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", 0.01, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("data_file", "./data/labeled_data_part",
                       "Data source for the  data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 64, "Dimensionality of character embedding (default: 128)")
#tf.flags.DEFINE_integer("seq_length", 600, "sequence length (default: 600)")
tf.flags.DEFINE_integer("num_classes", 5, "Number of classes (default: 5)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.8, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_integer("num_layers", 2, "number of layers (default: 2)")
tf.flags.DEFINE_integer("hidden_dim", 128, "neural numbers of hidden layer (default: 128)")
tf.flags.DEFINE_string("rnn_type", 'gru', "rnn type (default: gru)")
tf.flags.DEFINE_float("learning_rate", 1e-3, "learning rate (default:1e-3)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 5, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 500, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Data Preparation
# ==================================================

# Load data
print("Loading data...")
x_text, y = dataHelper.get_para_label(FLAGS.data_file)

# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_document_length)
# 神器，填充到最大长度
x = np.array(list(vocab_processor.fit_transform(x_text)))

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
# TODO: This is very crude, should use cross-validation
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

del x, y, x_shuffled, y_shuffled

print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


# Training
# ===============================================

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=0.5,
        allow_growth=True
    )
    session_config = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement,
        gpu_options=gpu_options
    )
    sess = tf.Session(config=session_config)
    with sess.as_default():
        rnn = TextRNN(
            sequence_length=x_train.shape[1],
            num_classes=FLAGS.num_classes,
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_dim=FLAGS.embedding_dim,
            num_layers=FLAGS.num_layers,
            hidden_dim=FLAGS.hidden_dim,
            rnn=FLAGS.rnn_type)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        grads_and_vars = optimizer.compute_gradients(rnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs-rnn", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", rnn.loss)
        acc_summary = tf.summary.scalar("accuracy", rnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        """
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
        """

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())


        def train_step(x_batch, y_batch):
            """
            A single training step
            :param x_batch:
            :param y_batch:
            :return:
            """
            feed_dict = {
                rnn.input_x: x_batch,
                rnn.input_y: y_batch,
                rnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }

            _, step, summaries, loss, accuracy = sess.run([train_op, global_step, train_summary_op, rnn.loss,
                                                           rnn.accuracy], feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)


        def dev_step(x_dev, y_dev):
            """
            Evaluates model on a dev set
            :param x_dev:
            :param y_dev:
            :return:
            """
            data_len = len(x_dev)
            batch_eval = dataHelper.batch_iter_eval(x_dev, y_dev)
            total_loss = 0.0
            total_acc = 0.0
            for x_batch_eval, y_batch_eval in batch_eval:
                batch_len = len(x_batch_eval)
                feed_dict = {
                    rnn.input_x: x_batch_eval,
                    rnn.input_y: y_batch_eval,
                    dropout_keep_prob: 1.0
                }
                loss, accuracy = sess.run(
                    [rnn.loss, rnn.accuracy],
                    feed_dict)
                total_loss += loss * batch_len
                total_acc += accuracy * batch_len
            time_str = datetime.datetime.now().isoformat()
            print("{}: loss {:g}, acc {:g}".format(time_str, total_loss / data_len, total_acc / data_len))


        # Generate batches
        batches = dataHelper.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))