# -*- coding: UTF-8 -*-
# @Time       : 2023/4/29 15:20
# @Author     : DYL
# @File       : forward_net.py
# @Description:

import numpy as np
import torch


class Sigmoid:
    def __init__(self):
        self.params = []

    @staticmethod
    def forward(x):
        return 1 / (1 + np.exp(-x))


class Affine:
    def __init__(self, W, b):
        self.params = [W, b]

    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        return out


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        # W1 = np.random.randn(I, H)
        W1 = torch.randn(I, H)
        # b1 = np.random.randn(H)
        b1 = torch.randn(H)
        # W2 = np.random.randn(H, O)
        W2 = torch.randn(H, O)
        # b2 = np.random.randn(O)
        b2 = torch.randn(O)

        # 生成层
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]

        # 权重添加到列表
        self.params = []
        for layer in self.layers:
            self.params += layer.params

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
