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
@file: config.py
@time: 17-5-11 下午9:50
'''
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf

flags = tf.flags
flags.DEFINE_integer("dimension", 384, "dimension of embedding")
flags.DEFINE_integer("units", 256, "the units of Hidden layer")
flags.DEFINE_float("l2", 0.0001, "the le regularization")
flags.DEFINE_float("dropRate", 0.1, "the drop rate")
flags.DEFINE_integer('batchSize', 32, "the batch size")
flags.DEFINE_integer("maxDocumentLength", 1319, "the largest number of word in document")
flags.DEFINE_integer("maxQueryLength", 210, "the largest number of word in query")
flags.DEFINE_integer("maxCandidate", 11, "the largest number of candidates ")
flags.DEFINE_integer("vocabSize", 67802, "vocabulary size")
flags.DEFINE_integer("trainSampleSize", 120769, "the train sample size")
FLAGS = flags.FLAGS