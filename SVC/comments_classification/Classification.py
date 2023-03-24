# -*- coding: UTF-8 -*-
# @Time       : 2023/3/20 14:06
# @Author     : DYL
# @File       : Classification.py
# @Description: PyCharm
import operator
import re

import jieba
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearnex import patch_sklearn


class ModelTrainingToClassification:  # 模型训练
    def __init__(self, filepath):
        """
        :param filepath: 数据集路路径
        """
        self.filepath = filepath
        self.sentenceID_categoryName = {}  # 评论ID和分类名的映射
        self.categoryID_categoryName = {}  # 分类ID和分类名的映射
        self.categoryID = []  # 分类ID和文本的映射
        self.sentence = []  # 分类ID和文本的映射
        self.training_text = None  # 训练集文本
        self.training_labels = None  # 训练集标签
        self.test_text = None  # 测试集文本
        self.test_labels = None  # 测试机标签
        self.predit = None  # 测试预测结果

    def divide_dataset(self, features, labels):
        """
        :param features: 向量化后的特征集
        :param labels: 特征集对应的分类标签集
        :return:
        """
        self.training_text, self.test_text, self.training_labels, self.test_labels = train_test_split(
            features, labels, test_size=0.2)  # 按八比二划分训练集和测试集

    @staticmethod
    def load_stopwords():
        """
        :return: 加载好的停用词列表
        """
        with open('../hit_stopwords.txt', 'r', encoding='UTF-8') as f:
            lines = f.readlines()
            stopwords = []
            for line in lines:
                stopwords.append(line.replace('\n', ''))
        return stopwords

    @staticmethod
    def remove_stopwords(word_list, stopwords):
        """
        :param word_list: 一行文本的分词结果列表
        :param stopwords: 停用词列表
        :return: 去掉停用词的文本分词结果字符串，以空格隔开
        """
        return " ".join(word for word in word_list if word not in stopwords)

    def preparement(self):
        """
        :return: 无，数据的预处理
        """
        with open(self.filepath, 'r', encoding='UTF-8') as f:
            article = f.readlines()
            stopwords = self.load_stopwords()
            for line in article:
                line_list = line.replace(' ', '').replace('\n', '').split('_!_')
                self.sentenceID_categoryName[line_list[0]] = line_list[2]  # 记录每条文本ID和分类名的对应关系
                if line_list[1] not in self.categoryID_categoryName:  # 记录分类ID和分类名之间的对应关系
                    self.categoryID_categoryName[line_list[1]] = line_list[2]
                self.categoryID.append(line_list[1])  # 记录每行文本的对应ID
                sentence = ''
                rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")  # 利用正则过滤掉每行文本中除了汉字、字母、数字之外的符号
                for i in range(3, len(line_list)):
                    sentence += rule.sub('', line_list[i])
                words = self.remove_stopwords(jieba.cut(sentence), stopwords)  # 去停用词
                self.sentence.append(words)  # 记录经过处理之后的每行文本

    def tdf_idf(self):
        """
        :return: 返回整个数据集特征向量化后的结果，对每行预处理之后的文本选出特征并做向量化
        """
        tfidf = TfidfVectorizer(norm='l2', ngram_range=(1, 2))  # 为保证特征的丰富性和训练的效果，除了分词之后的每个单独的字词用作特征外，还将相邻的两个词组用作特征
        return tfidf.fit_transform(self.sentence), self.categoryID

    def model_training(self):
        """
        :return: 无，使用支持向量机模型LinearSVC()进行模型训练
        """
        model = LinearSVC()
        model.fit(self.training_text, self.training_labels)  # 线性拟合
        return model

    def model_predit(self, model):
        self.predit = model.predict(self.test_text)
        conf_mat = confusion_matrix(self.test_labels, self.predit)
        _, _ = plt.subplots(figsize=(15, 12))
        self.categoryID_categoryName = dict(sorted(self.categoryID_categoryName.items(), key=operator.itemgetter(0)))
        sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=self.categoryID_categoryName.values(),
                    yticklabels=self.categoryID_categoryName.values())  # 模型的混淆矩阵构建
        plt.ylabel('实际结果', fontsize=18)
        plt.xlabel('预测结果', fontsize=18)
        plt.show()
        print(self.predit.shape[0])  # 测试用例的数量
        print('accuracy %s' % accuracy_score(self.predit, self.test_labels))  # 精确度
        # 输出模型评估报告
        print(classification_report(self.test_labels, self.predit, target_names=self.categoryID_categoryName.keys()))

    def main(self):
        self.preparement()  # 数据预处理
        features, labels = self.tdf_idf()  # 特征向量化
        self.divide_dataset(features, labels)  # 划分数据集
        model = self.model_training()  # 模型训练
        self.model_predit(model)  # 模型测试


if __name__ == '__main__':  # 差不多要跑两分多钟
    patch_sklearn()  # 英特尔的一个关于sklearn相关算法的加速包
    M = ModelTrainingToClassification('../comments/toutiao_cat_data.txt')
    M.main()
