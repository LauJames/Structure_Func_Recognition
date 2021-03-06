#! /user/bin/evn python
# -*- coding:utf8 -*-

"""

@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : StructureFuncRecognition 
@File     : eval.py.py
@Time     : 2017/12/27 22:57
@Software : PyCharm
@Copyright: "Copyright (c) 2017 Lau James. All Rights Reserved"
"""

import tensorflow as tf
import numpy as np
import os
import csv
from data import dataHelper
from tensorflow.contrib import learn
from sklearn import metrics

# Parameters
# 运行时传参：./eval.py eval_train checkpoint_dir="./runs/1516450545/checkpoints/"
# ====================================

# Data Parameters
tf.flags.DEFINE_string("data_file", "./data/labeled_data", "Data source")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1516450545/checkpoints/", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{} = {}".format(attr.upper(), value))
print("")

# CHANGE THIS: Load data. Load own data here
if FLAGS.eval_train:
    x_raw, y_test = dataHelper.get_para_label(FLAGS.data_file)
    y_test = np.argmax(y_test, axis=1)
else:
    pass

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, '..', "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

# Evaluation
# =================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement
    )
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]
        # 注意后面为0表示第一个为这个名字的，可能出现多个同名，tensorflow会根据0/1/2/3来避免冲突

        # Generate batches for on epoch
        batches = dataHelper.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, feed_dict={input_x: x_test_batch, dropout_keep_prob:1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))
    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test, all_predictions)
    print(cm)

# Save the evaluation to a csv
# predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
# out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
# print("Saving evaluation to {0}".format(out_path))
# with open(out_path, 'w') as f:
#     csv.writer(f).writerows(predictions_human_readable)
