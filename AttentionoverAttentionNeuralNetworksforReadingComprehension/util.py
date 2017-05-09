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
@file: util.py
@time: 17-5-3 上午9:20
'''
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import tensorflow as tf


def softmax(_datas, _axis, _mask):

    maxAxis = tf.reduce_max(_datas, axis=_axis, keep_dims=True)
    _datas = _datas - maxAxis
    maskDatas = tf.exp(_datas)*_mask
    sumDatas = tf.reduce_sum(maskDatas, axis=_axis, keep_dims=True)
    # 此处应该不要为结果加一个小数，防止除以0,感觉被掩盖的数的原始值不太可能远大于没被掩盖的数
    # 所以此处应该不会有0
    softmaxValue = maskDatas / sumDatas

    return softmaxValue


def test_softmax():
    import numpy as np
    a = np.arange(12, dtype=np.float32).reshape(2,2,3)
    print(a)
    b = np.concatenate((a, np.zeros((2,2,2))), axis=2)
    print(b)
    mask = np.concatenate((np.ones((2,2,3)), np.zeros((2,2,2))), axis=2)
    print(mask)

    data = tf.Variable(a)
    expandData = tf.Variable(b)
    mask = tf.Variable(mask)

    expandDataSoftmax = softmax(expandData, 2, mask)
    dataSoftmax = tf.nn.softmax(data, dim=2)
    # 此处的0,1都好使，使用-1可以表示最后一维，但无法使用2, 好像是这个版本的bug
    # t = tf.reduce_mean(data, 2)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    d2 = sess.run(dataSoftmax)
    d  = sess.run(expandDataSoftmax)


    print(d)
    print(d2)


if __name__ == "__main__":
    test_softmax()