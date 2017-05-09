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
@file: reader.py.py
@time: 17-5-1 上午10:06
'''
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import numpy as np
import tensorflow as tf


def explore_data():
    path = 'data/cbtest_NE_train.txt'
    i = 0
    with open(path) as f:
        while f:
           print(i)
           i += 1
           sign = extract_one_sample(f)
           if sign==1:
               break


def build_data_matrix(_path):
    '''
    返回由文档、查询、回答和候选答案的列表。因为原始数据的格式是：
    1. 文档句子1
    2. 文档句子2
    ...
    20. 文档句子20
    21. query  answer  candidates
    这种格式，本函数就是从这种函数中提取出需要的结构信息
    :param _path:
    :return:document、query以及answer的都是用空格隔开的字符串表示，candidate是字符串的列表表示的
    '''
    documents, querys, answers, candidates  = ([] for _ in range(4))
    # 上面的代码不能用([],)*4来代替，若用([],)*4来代替则会这四个变量表示的是同一个[]对象
    with open(_path) as f:
        while f:
            sign = extract_one_sample(f)
            if sign==1:
                break
            else:
                documents.append(sign[0])
                querys.append(sign[1])
                answers.append(sign[2])
                candidates.append(sign[3])
                # print(sign[0])
                # break
    # print(len(querys))
    # print(querys[0])
    # build_word_to_indice_map(documents)
    return documents, querys, answers, candidates


def build_word_to_indice_map(_documents):
    '''
    建立词到索引的map,第0号索引留给填充位，1号索引留给未在该map中出现的词
    此处建立的词汇表与论文里的统计数据不一致
    :param _documents:
    :return:
    '''
    wordToIndice = {}
    i = 2

    for d in _documents:
        words = d.split()
        for word in words:
            if not wordToIndice.has_key(word):
                wordToIndice[word] = i
                i += 1
    print('vocabulary size is ')
    print(len(set(wordToIndice.keys())))
    return wordToIndice


def max_word_in_list(_datas):
    '''
    _datas的元素为列表则返回列表的最大长度，如果是空格隔开的字符串则返回最大词数
    :param _datas:
    :return:
    '''
    maxLength = 0
    for data in _datas:
        if( isinstance(data, list) == True ):
            length = len(data)
        else:
            length = len(data.split())

        if length>maxLength:
            maxLength = length

    return maxLength


def build_indice_matrix(_datas, _wordToIndice, _maxLength):
    '''
    通过词的列表建立一个矩阵，并且每个句子都填充到_maxLength长,长于_maxLength的部分则截断
    :param _datas:
    :param _wordToIndice:
    :param _maxLength:
    :return:
    '''
    sampleNums = len(_datas)
    sign = isinstance(_datas[0], list)
    dataMatrix = []
    sequenceLength = []
    if not sign:
        for i in xrange(sampleNums):
            words = _datas[i].split()
            length = len(words)
            sequenceLength.append(length)
            gap = _maxLength - length

            oneRow = []
            for word in words:
                if(_wordToIndice.has_key(word) ):
                    oneRow.append(_wordToIndice[word])
                else:
                    oneRow.append(1)

            oneRow.extend([0]*gap)   #若gap为负数，则[0]*gap为[]
            oneRow = oneRow[:_maxLength]
            dataMatrix.append(oneRow)
    else:
        for i in xrange(sampleNums):
            length = len(_datas[i])
            sequenceLength.append(length)
            gap = _maxLength - length

            oneRow = []
            for word in _datas[i]:
                if (_wordToIndice.has_key(word)):
                    oneRow.append(_wordToIndice[word])
                else:
                    oneRow.append(0)

            oneRow.extend([0] * gap)
            oneRow = oneRow[:_maxLength]
            dataMatrix.append(oneRow)

    return dataMatrix, sequenceLength


def build_document_word_flag_list(_document, _word):
    flags = [0]*len(_document)
    for i in xrange(len(_document)):
        if _document[i] == _word:
            # flags.append(i)
            flags[i] = 1
    return flags


def build_document_candidate_matrix(_documents, _candidates):
    matrix = []
    i= 0
    for d, candidate in zip(_documents, _candidates):
        i += 1
        if(i%100 == 0):
            print(i)

        dmatrix = []  #每行存放每个候选词在document中的指示
        for c in candidate:
            # print(c)
            # if c!=0:
                dmatrix.append(build_document_word_flag_list(d, c))
        matrix.append(dmatrix)

    return matrix

def build_train_matrix():
    documents, querys, answers, candidates = build_data_matrix('data/cbtest_NE_train.txt')
    wordToIndice, maxLength = build_word_to_indice_map(documents)
    datas, sequenceLength = build_indice_matrix(candidates, wordToIndice, maxLength)
    d = np.array(datas)
    print("result is ")
    print(d.shape)
    print(len(sequenceLength))
    print(sequenceLength)
    return datas, sequenceLength


def extract_one_sample(_f):
    '''
    返回的document, query,answer都是由空格隔开的字符串， candidate返回的是一个字符串列表
    :param _f:
    :return:
    '''
    data = ""
    for i in xrange(20):
        line = _f.readline()
        if line=="":
            return 1
        data += " ".join(line.split()[1:])
        # print(line)
    # print(data)

    line = _f.readline()
    datas = line.split()
    query = " ".join(datas[1:-2])
    answer = datas[-2]
    candidates = datas[-1].split('|')
    # print(data)
    # print(query)
    # print(answer)
    # print(candidates)
    _f.readline()
    return data, query, answer, candidates


def prepare_one_batch(_batchSize):
    documents, querys, answers, candidates = build_data_matrix('data/cbtest_NE_train.txt')
    wordToIndice, maxLength = build_word_to_indice_map(documents)
    datas, sequenceLength = build_indice_matrix(candidates, wordToIndice, maxLength)
    d = np.array(datas)
    print(d.shape)


def batch_data(_batchSize):
    pass


def test_queue():

    i = tf.train.range_input_producer(5, shuffle=True).dequeue()
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)
    for t in xrange(11):
        a = sess.run(i)
        print(a)


def data_producer(_rawData, _batchSize, ):
    raw_data = tf.convert_to_tensor(_rawData, name="raw_data", dtype=tf.int32)
    data_len = tf.shape(raw_data)[0]
    limit = data_len // _batchSize
    i = tf.train.range_input_producer(limit, shuffle=True).dequeue()

    return raw_data[i*_batchSize:(i+1)*_batchSize, :]


def test():
    documents, querys, answers, candidates = build_data_matrix('data/cbtest_NE_train.txt')
    wordToIndice, maxLength = build_word_to_indice_map(documents)
    datas, sequenceLength = build_indice_matrix(documents, wordToIndice, maxLength)
    d = np.array(datas)
    print(d.shape)
    data = data_producer(d, 5)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)
    for t in xrange(11):
        a = sess.run(data)
        print(a)

if __name__ == "__main__":
    # explore_data()
    # build_train_matrix()
    pass
    # prepare_one_batch(10)
    # test_queue()
    # data_producer(np.arange(10), 3)
    test()