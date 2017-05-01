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
    print(len(querys))
    print(querys[0])
    # build_word_to_indice_map(documents)
    return documents, querys, answers, candidates


def build_word_to_indice_map(_documents):
    wordToIndice = {}
    i = 1
    for d in _documents:
        words = d.split()
        for word in words:
            if not wordToIndice.has_key(word):
                wordToIndice[word] = i
                i += 1
    print( len(wordToIndice) )
    print( wordToIndice[wordToIndice.keys()[1]])
    return wordToIndice


def build_indice_matrix(_datas, _wordToIndice):
    sampleNums = len(_datas)
    sign = len(_datas[0])
    dataMatrix = []
    if sign==1:
        for i in xrange(sampleNums):
            words = _datas[i].split()
            oneRow = []
            for word in words:
                if(_wordToIndice.has_key(word) ):
                    oneRow.append(_wordToIndice[word])
                else:
                    oneRow.append(0)
            dataMatrix.append(oneRow)
    else:
        for i in xrange(sampleNums):
            oneRow = []
            for word in _datas[i]:
                if (_wordToIndice.has_key(word)):
                    oneRow.append(_wordToIndice[word])
                else:
                    oneRow.append(0)
            dataMatrix.append(oneRow)

    print(len(dataMatrix))
    print(len(dataMatrix[0]))

def build_train_matrix():
    documents, querys, answers, candidates = build_data_matrix('data/cbtest_NE_train.txt')
    wordToIndice = build_word_to_indice_map(documents)
    build_indice_matrix(candidates, wordToIndice)


def extract_one_sample(_f):
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





if __name__ == "__main__":
    # explore_data()
    build_train_matrix()