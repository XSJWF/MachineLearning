# -*- coding: UTF-8 -*-
# @Time       : 2023/4/15 9:41
# @Author     : DYL
# @File       : forward_net.py
# @Description: 前向传播实现
import numpy as np


class Sigmoid:
    def __init__(self):
        self.params = []

    @staticmethod
    def forward(x):
        return 1 / (1 + np.exp(-x))


class Affine:
    def __init__(self, w, b):
        self.params = [w, b]

    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        return out

    class TwoLayerNet:
        def __init__(self, input_size, hidden_size, output_size):
            I, H, O = input_size, hidden_size, output_size

            # 初始化权重和偏置
            W1 = np.random.rand(I, H)
            b1 = np.random.randn(H)
            W2 = np.random.randn(H, O)
            b2 = np.random.randn(O)

            # 生成层
            self.layers = [
                Affine(W1, b1),
                Sigmoid(),
                Affine(W2, b2)
            ]

            # 将所有的权重整理到列表中
            self.params = []
            for layer in self.layers:
                self.params += layer.params

        def predict(self, x):
            for layer in self.layers:
                x = layer.forward(x)
            return x
