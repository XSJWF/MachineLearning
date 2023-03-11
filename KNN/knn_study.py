# -*- coding: UTF-8 -*-
# @Time    : 2023/2/22 19:34
# @Author  : 溪风
# @File    : knn_study.py
# @Description :K-邻近算法简单实现

from numpy import *
import operator

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, K):
    """
    :param inX: 目标特征集
    :param dataSet: 特征数据集
    :param labels: 对应特征数据集的标签数据集
    :param K: 最近邻K的k值
    :return: 返回目标特征集经过k-近邻算法所得出的分类结果
    """
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndices = distances.argsort()
    classCount = {}
    for i in range(K):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
