# -*- coding: UTF-8 -*-
# @Time    : 2023/2/22 19:39
# @Author  : 溪风
# @File    : knn_study_main.py
# @Description :

from KNN import knn_study

group, labels = knn_study.createDataSet()
print(knn_study.classify0([0, 0], group, labels, 3))
