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
flags.DEFINE_integer("dimension", 384, "dimension of embedding")
flags.DEFINE_integer("units", 256, "the units of Hidden layer")
flags.DEFINE_float("l2", 0.0001, "the le regularization")
flags.DEFINE_float("dropRate", 0.1, "the drop rate")
flags.DEFINE_integer('batchSize', 5, "the batch size")
flags.DEFINE_integer("maxDocumentLength", 1319, "the largest number of word in document")
flags.DEFINE_integer("maxQueryLength", 210, "the largest number of word in query")
flags.DEFINE_integer("maxCandidate", 11, "the largest number of candidates ")
FLAGS = flags.FLAGS



def get_embedding(_shape):
    embeddings = tf.Variable(
        tf.random_uniform(_shape, -0.05, 0.05)
    )
    return embeddings


def build_biGRU(_inputs, _sequenceLength):
    gruCellForward = tf.contrib.rnn.core_rnn_cell.GRUCell(FLAGS.units)
    gruCellBackward = tf.contrib.rnn.core_rnn_cell.GRUCell(FLAGS.units)
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


class AOAInput(object):

    def __init__(self, _documents, _querys, _answers, _candidates, _sequencesLength):
        self.documents = _documents
        self.querys = _querys
        self.answers = _answers
        self.candidates = _candidates
        self.sequencesLength = _sequencesLength


class AOAInputByIndice(object):

    def __init__(self, _path):
        datas = build_data_matrix(_path)
        self.wordIndiceMap = build_word_to_indice_map(datas[0])
        self.documents = []
        self.querys = []
        self.answers = []
        self.candidates = []
        self.indice = 0
        self.epochSize = 0
        self.indiceCounter = 0
        # documents, documentsLength = build_indice_matrix(datas[0], self.wordIndiceMap, FLAGS.maxDocumentLength)
        # querys, queryLength = build_indice_matrix(datas[1], self.wordIndiceMap, FLAGS.maxQueryLength)
        # candidates, candidatesLength = build_indice_matrix(datas[3], self.wordIndiceMap, FLAGS.maxCandidate)
        #之前的最大长度正确是因为有build_indice_matrix得到的最大长度，而不是使用修改之前的max_word_in_list函数
        #该函数之前由于使用if(len(data)>1)判断是否是列表的逻辑不正确，因为字符串的长度大于1
        documents, documentsLength = build_indice_matrix(datas[0], self.wordIndiceMap, FLAGS.maxDocumentLength)
        querys, queryLength = build_indice_matrix(datas[1], self.wordIndiceMap, FLAGS.maxQueryLength)
        candidates, candidateLength = build_indice_matrix(datas[3], self.wordIndiceMap, FLAGS.maxCandidate)
        answers, _ = build_indice_matrix(datas[2], self.wordIndiceMap, 1)
        self.documents = documents
        self.querys = querys
        self.answers = answers
        self.candidates = candidates
        self.documentLength = documentsLength
        self.queryLength = queryLength
        self.candidateLength = candidateLength
        print (len(documents))
        print (documents[0])
        print (documentsLength[0])
        # print (len(querys[0]))
        # print (candidates[0])
        print (len(querys[0]))
        print(querys[0])
        print (candidates[0])
        print (answers[0])
        print( build_document_word_flag_list(documents[0], answers[0][0]))
        # for j in xrange(len(documents)):
        #     for i in xrange(11):
        #         length = sum(build_document_word_flag_list(documents[j], candidates[j][i]))
        #         if(length>1):
        #             print('ye')
        #             print(length)

        matrix = build_document_candidate_matrix(documents, candidates)
        matrix = np.array(matrix)
        print(len(matrix))
        print(len(matrix[0]))
        print(len(matrix[0][0]))
        print(matrix.shape)


class AOAModel(object):
    #现在的sequenceLength还存在问题，因为document和query的肯定不同
    def __init__(self, _isTraining, _input):
        self.keepProb = 1 - FLAGS.dropRate
        self.embeddings = tf.Variable(tf.random_uniform([FLAGS.vocabSize, FLAGS.dimension], -0.05, 0.05) )

        def gru_cell():
            return tf.contrib.rnn.core_rnn_cell.GRUCell(FLAGS.units)
        attn_cell = gru_cell

        if _isTraining and self.keepProb<1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(gru_cell(), output_keep_prob=self.keepProb)

        self.gruCellForward = attn_cell()
        self.gruCellBackward = attn_cell()
        self.initialStateForward = self.gruCellForward.zero_state(FLAGS.batchSize, tf.float32)
        self.initialStateBackward = self.gruCellBackward.zero_state(FLAGS.batchSize, tf.float32)

        (_, outputsStateForDocument) = tf.nn.bidirectional_dynamic_rnn(self.gruCellForward,
                                                                       self.gruCellBackward,
                                                                       inputs=_input.documents,
                                                                       sequence_length=_input.sequencesLength,
                                                                       initial_state_fw=self.initialStateForward,
                                                                       initial_state_bw=self.initialStateBackward)
        hiddenDocument = tf.concat(outputsStateForDocument, 2)

        self.gruCellForwardForQuery = attn_cell()
        self.gruCellBackwardForQuery = attn_cell()
        self.initialStateForwardForQuery = self.gruCellForwardForQuery.zero_state(FLAGS.batchSize, tf.float32)
        self.initialStateBackwardForQuery = self.gruCellBackwardForQuery.zero_state(FLAGS.batchSize, tf.float32)

        (_, outputsStateForQuery) = tf.nn.bidirectional_dynamic_rnn(self.gruCellForwardForQuery,
                                                                    self.gruCellBackwardForQuery,
                                                                    inputs=_input.querys,
                                                                    sequence_length=_input.sequencesLength,
                                                                    initial_state_fw=self.initialStateForwardForQuery,
                                                                    initial_state_bw=self.initialStateBackwardForQuery)
        hiddenQuery = tf.concat(outputsStateForQuery, 2)  #可能出现问题
        hiddenQueryT = tf.matrix_transpose(hiddenQuery)
        scoreMatrix = tf.matmul(hiddenDocument, hiddenQueryT) #shape为 batch×document*query

        columnWiseSoftmax = tf.nn.softmax(scoreMatrix)

        rowWiseSoftmax = tf.nn.softmax(scoreMatrix, dim=1)
        columWeight = tf.reduce_mean(rowWiseSoftmax, axis=1, keep_dims=True)

        columWeightT = tf.matrix_transpose(columWeight)
        result = tf.matmul(columnWiseSoftmax, columWeightT)




if __name__ == "__main__":
    # test()
    # print (FLAGS.dimension)
    input = AOAInputByIndice('data/cbtest_NE_train.txt')