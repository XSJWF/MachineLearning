# -*- coding: UTF-8 -*-
# @Time       : 2023/3/20 14:06
# @Author     : DYL
# @File       : Classification.py
# @Description: PyCharm
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearnex import patch_sklearn

from SVC.comments_classification.Classification import ModelTrainingToClassification


class ModelOpt:  # 模型选择
    def __init__(self, filepath):
        self.training_labels = None  # 训练集分类标签
        self.training_text = None  # 训练集文本
        self.path = filepath  # 数据集路径

    def model_option(self):  # 多种模型测试验证
        models = [
            RandomForestClassifier(n_estimators=100, max_depth=3, random_state=0),  # 随机森林
            LinearSVC(),  # 支持向量机——线性分类
            MultinomialNB(),  # 多项式朴素贝叶斯
            LogisticRegression(solver='liblinear', random_state=0)  # 逻辑回归
        ]
        cv = 10  # k折交叉验证的折数
        entries = []
        for model in models:
            model_name = model.__class__.__name__  # 获取当前模型的名字
            print(model_name)
            # 计算每种模型的精确度
            accuracies = cross_val_score(model, self.training_text, self.training_labels, scoring='accuracy', cv=cv)
            for fold_idx, accuracy in enumerate(accuracies):
                entries.append((model_name, fold_idx, accuracy))
        cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
        # 构建一幅箱线图来对比各个模型的效果
        sns.boxplot(x='model_name', y='accuracy', data=cv_df)
        sns.stripplot(x='model_name', y='accuracy', data=cv_df,
                      size=8, jitter=True, edgecolor="gray", linewidth=2)
        plt.show()
        print(cv_df.groupby('model_name').accuracy.mean())

    def main(self):
        classification = ModelTrainingToClassification(self.path)  # 初始化一个模型训练的对象
        classification.preparement()  # 数据预处理
        features, labels = classification.tdf_idf()  # 训练数据特征向量化
        classification.divide_dataset(features, labels)  # 分出训练集来测试模型的好坏
        # 获取训练集的文本和标签
        self.training_text, self.training_labels = classification.training_text, classification.training_labels
        self.model_option()  # 模型对比选择


if __name__ == '__main__':  # 这个多种训练模型对比的算法要跑比较久，差不多半小时
    patch_sklearn()  # sklearn相关机器学习算法的运行计算加速
    M = ModelOpt('comments/toutiao_cat_data.txt')
    M.main()
