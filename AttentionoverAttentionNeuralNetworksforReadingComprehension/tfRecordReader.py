# --*--coding:utf8--*--

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from reader import *




def produce_recorder_data_file(_path):
    writer = tf.python_io.TFRecordWriter(_path)

    # datas = build_data_matrix('data/cbtest_NE_train.txt')
    # wordIndiceMap = build_word_to_indice_map(datas[0])
    # documents = build_indice_matrix(datas[0], wordIndiceMap, 1319)
    # querys = build_indice_matrix(datas[1], wordIndiceMap, 210)
    # answers = build_indice_matrix(datas[2], wordIndiceMap, 1)
    documents = np.arange(20).reshape((5,4))
    querys = np.arange(15).reshape((5,3))
    answers = np.arange(5).reshape((5,1))
    for i in xrange(len(documents)):
        documentsFeature = tf.train.Feature(int64_list=tf.train.Int64List(value=documents[i]))
        querysFeature = tf.train.Feature(int64_list=tf.train.Int64List(value=querys[i]))
        answersFeature = tf.train.Feature(int64_list=tf.train.Int64List(value=answers[i]))

        features = tf.train.Features(feature={'documents':documentsFeature, 'querys':querysFeature, 'answers':answersFeature})
        example = tf.train.Example(features=features)
        writer.write(example.SerializeToString())
    writer.close()


def read_tfrecord():
    path = ["train.tf"]
    reader = tf.TFRecordReader()
    inputs = tf.train.string_input_producer(path)
    _, example = reader.read(inputs)
    features = tf.parse_single_example(example, features={
        'documents':tf.FixedLenFeature([4], tf.int64),
        'querys':tf.FixedLenFeature([3], tf.int64),
        'answers':tf.FixedLenFeature([1], tf.int64),
    })

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # tf.train.start_queue_runners(sess=sess)
    tf.train.start_queue_runners(sess=sess)
    f = sess.run(features)
    print(f['documents'])


if __name__ == "__main__":
    # produce_recorder_data_file("train.tf")
    read_tfrecord()