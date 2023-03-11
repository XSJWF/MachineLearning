# -*- coding: UTF-8 -*-
# @Time    : 2023/2/22 21:15
# @Author  : 溪风
# @File    : knn_1.py
# @Description :knn算法现实应用实现，改进约会网站

import matplotlib.pyplot as plt
from numpy import zeros, array, tile

import knn_study


def file2matrix(filename):
    """
    :param filename: 数据集的地址
    :return: 返回一个特征集和一个分类标签集
    """
    with open(filename, 'r', encoding='utf-8') as f:
        arraylines = f.readlines()
    returnmat = zeros((len(arraylines), 3))
    labelsvector = []
    index = 0
    for line in arraylines:
        features_label = line.replace("\n", "").split('\t')
        returnmat[index, :] = features_label[:3]
        if features_label[-1] == 'largeDoses':
            labelsvector.append(int(3))
        elif features_label[-1] == 'smallDoses':
            labelsvector.append(int(2))
        else:
            labelsvector.append(int(1))
        index += 1
    return returnmat, labelsvector


def autonorm(dataset):
    """
    :param dataset: 特征集
    :return: 返回经过标准归一化之后的特征集和每个特征的最大最小差值集合,和每个特征最小值的集合
    """
    minvals = dataset.min(0)
    maxvals = dataset.max(0)
    ranges = maxvals - minvals
    m = dataset.shape[0]
    normdataset = dataset - tile(minvals, (m, 1))
    normdataset = normdataset / tile(ranges, (m, 1))
    return normdataset, ranges, minvals


def datingclasstest():
    """
    :return: 对算法模型进行测试,无返回值
    """
    testradio = 0.1
    datingdatamat, datingdatalabels = file2matrix("datingTestSet.txt")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingdatamat[:, 1], datingdatamat[:, 2], 15.0 * array(datingdatalabels), 15.0 * array(datingdatalabels))
    plt.show()
    normmat, ranges, minvals = autonorm(datingdatamat)
    m = normmat.shape[0]
    testnum = int(m * testradio)
    error = 0
    for i in range(testnum):
        predict_result = knn_study.classify0(normmat[i, :], normmat[testnum:m, :], datingdatalabels[testnum:m], 3)
        print(f"第{i}条数据的预测结果为: ", predict_result, ",其真实结果为: ", datingdatalabels[i])
        if predict_result != datingdatalabels[i]:
            error += 1
    print("模型预测的准确率为: ", (m - error) / m)


def classifyperson():
    """
    :return: 算法的具体简单命令应用,无返回值
    """
    resultset = ['didntLike', 'smallDoses', 'largeDoses']
    datingdatamat, datingdatalabels = file2matrix("datingTestSet.txt")
    normmat, ranges, minvals = autonorm(datingdatamat)
    features = zeros(3)
    features[0] += float(input("请输入他(她)每年获得的飞行常客里程数: "))
    features[1] += float(input("请输入他(她)玩视频游戏所耗时间百分比: "))
    features[2] += float(input("请输入他(她)每周消费的冰淇淋公升数: "))
    result = knn_study.classify0((features - minvals) / ranges, normmat, datingdatalabels, 3)
    print("该人可能属于 ", resultset[result - 1], " 类型")


if __name__ == '__main__':
    datingclasstest()  # 测试算法的好坏
    # classifyPerson()  # 算法的应用
