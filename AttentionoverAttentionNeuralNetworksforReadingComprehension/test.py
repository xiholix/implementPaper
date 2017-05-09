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
@file: test.py
@time: 17-5-3 下午6:43
'''
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import tensorflow as tf


def testSparseTensor():
    a2 = tf.SparseTensor(indices=[[0,1], [0,3], [1,2], [1,3], [2,0]], values=[1,1,1,1,1], dense_shape=[3,4])
    b = tf.Variable([[1], [2], [3], [4]])
    # b = tf.Variable([[1,2,3,4]])
    # a2 = tf.sparse_tensor_to_dense(a2, 0)
    # c = tf.matmul(a2,b)
    c = tf.sparse_tensor_dense_matmul(a2, b)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    result = sess.run(c)
    print(result)


def testEmbeddingLookup():
    pass


if __name__ == "__main__":
    testSparseTensor()