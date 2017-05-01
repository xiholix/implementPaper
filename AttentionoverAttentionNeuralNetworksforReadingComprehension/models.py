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
@file: models.py
@time: 17-5-1 下午3:16
'''
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import tensorflow as tf

from reader import  *

flags = tf.flags
flags.DEFINE_integer("dimension", 384, "dimension of GRUs")
flags.DEFINE_float("l2", 0.0001, "the le regularization")
flags.DEFINE_float("dropRate", 0.1, "the drop rate")
flags.DEFINE_integer('batchSize', 5, "the batch size")
FLAGS = flags.FLAGS



def get_embedding(_shape):
    embeddings = tf.Variable(
        tf.random_uniform(_shape, -0.05, 0.05)
    )
    return embeddings


def build_biGRU(_inputs, _sequenceLength):
    gruCellForward = tf.contrib.rnn.core_rnn_cell.GRUCell(FLAGS.dimension)
    gruCellBackward = tf.contrib.rnn.core_rnn_cell.GRUCell(FLAGS.dimension)
    initialStateForward = gruCellForward.zero_state(FLAGS.batchSize, tf.float32)
    initialStateBackward = gruCellBackward.zero_state(FLAGS.batchSize, tf.float32)

    (outputs, outputsState) = tf.nn.bidirectional_dynamic_rnn(gruCellForward,
                                                       gruCellBackward,
                                                       inputs=_inputs,
                                                       sequence_length=_sequenceLength,
                                                       initial_state_fw=initialStateForward,
                                                       initial_state_bw=initialStateBackward)

def test():
    embeddins = get_embedding([67800,FLAGS.dimension])
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    em = sess.run(embeddins)
    print(em)
    datas = build_train_matrix()
    em = tf.nn.embedding_lookup(embeddins, datas[0])
    em = sess.run(em)
    print(em)
    print(em.shape)

if __name__ == "__main__":
    test()
    print (FLAGS.dimension)