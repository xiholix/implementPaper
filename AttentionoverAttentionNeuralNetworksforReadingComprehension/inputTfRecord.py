# -*-coding:utf8-*-
#                       _oo0oo_
#                      o8888888o
#                      88" . "88
#                      (| -_- |)
#                      0\  =  /0
#                    ___/`---'\___
#                  .' \\|     |// '.
#                 / \\|||  :  |||// \
#                / _||||| -:- |||||- \
#               |   | \\\  -  /// |   |
#               | \_|  ''\---/''  |_/ |
#               \  .-\__  '-'  ___/-. /
#             ___'. .'  /--.--\  `. .'___
#          ."" '<  `.___\_<|>_/___.' >' "".
#         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
#         \  \ `_.   \_ __\ /__ _/   .-` /  /
#     =====`-.____`.___ \_____/___.-`___.-'=====
#                       `=---='
#
#
#     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#               佛祖保佑         永无BUG
#
#
'''
@version: ??
@author: xiholix
@contact: x123872842@163.com
@software: PyCharm
@file: inputTfRecord.py
@time: 17-5-9 下午6:25
'''
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import numpy as np

from reader import *
# from models import FLAGS


def writeTfRecorder(_path):

    writer = tf.python_io.TFRecordWriter(_path)
    datas = build_data_matrix('data/cbtest_NE_train.txt')
    wordIndiceMap = build_word_to_indice_map(datas[0])
    documents, documentLength = build_indice_matrix(datas[0], wordIndiceMap, 1319)
    querys, queryLength = build_indice_matrix(datas[1], wordIndiceMap, 210)
    answers, answerLength = build_indice_matrix(datas[2], wordIndiceMap, 1)

    print( len(answers))
    for i in xrange(len(querys)):
        documentsFeature = tf.train.Feature(int64_list=tf.train.Int64List(value=documents[i]))
        querysFeature = tf.train.Feature(int64_list=tf.train.Int64List(value=querys[i]))
        answersFeature = tf.train.Feature(int64_list=tf.train.Int64List(value=answers[i]))
        documentLengthFeature = tf.train.Feature(int64_list=tf.train.Int64List(value=[documentLength[i]]))
        queryLengthFeature = tf.train.Feature(int64_list=tf.train.Int64List(value=[queryLength[i]]))

        features = tf.train.Features(feature={
            'document':documentsFeature,
            'query':querysFeature,
            'answer':answersFeature,
            'documentLength':documentLengthFeature,
            'queryLength':queryLengthFeature
        })

        example = tf.train.Example(features=features)
        writer.write(example.SerializeToString())

    writer.close()


def reader_tfrecorder():
    path = ['tfRecord/train.tfrecord']
    queue = tf.train.string_input_producer(path)
    reader = tf.TFRecordReader()
    _, serializeFeatures = reader.read(queue)
    examples = tf.parse_single_example(serializeFeatures, features={
        'document':tf.FixedLenFeature([1319], tf.int64),
        'query':tf.FixedLenFeature([210], tf.int64),
        'answer':tf.FixedLenFeature([], tf.int64),
        'documentLength':tf.FixedLenFeature([], tf.int64),
        'queryLength':tf.FixedLenFeature([], tf.int64)
    })

    batchData = tf.train.shuffle_batch([examples['document'], examples['query'], examples['answer'],
                                        examples['documentLength'], examples['queryLength']], 5, 25, 5)
    # shuffle_batch的结果的batch内容存在第一维中，第0维的长度是第一个参数列表的长度


    # init = tf.global_variables_initializer()
    # sess = tf.Session()
    # sess.run(init)
    # tf.train.start_queue_runners(sess=sess)
    #
    # example = sess.run(examples)
    # print((example['answer']))
    # print((example['documentLength']))
    # print(example['queryLength'])

    return batchData

if __name__ == "__main__":
    writeTfRecorder("tfRecord/train.tfrecord")
    # reader_tfrecorder()