#!/usr/bin/python3
# -*- encoding: utf-8 -*-
"""
@Project Name :   BP
@File         :   BP_NN.py
@Contact      :   s1142233286@163.com
@License      :   (C)Copyright 2018-2019

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/6/25 10:01    sunan    1.0         None
"""

# import tensorflow as tf
# import lib
# 输入数据
import numpy as np
X = np.array([[1, 0, 0],
              [1, 0, 1],
              [1, 1, 0],
              [1, 1, 1]])
# 标签
Y = np.array([[0, 1, 1, 0]])
# 权重初始化，1列3行，取值范围-1,到1
V = np.random.random((3, 4)) * 2 - 1
W = np.random.random((4, 1)) * 2 - 1
print("V : \n", V)
print("W : \n", W)
# 学习率设置
lr = 0.11
# 计算迭代次数
n = 0
# 神经网络输出
o = 0


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def disigmoid(x):
    return x*(1 - x)


def update():
    global X, Y, W, V, lr

    L1 = sigmoid(np.dot(X, V))  # 隐藏层输出（4,4）
    L2 = sigmoid(np.dot(L1, W))  # 输出层输出（4,1）
    L2_delta = (Y.T - L2) * disigmoid(L2)
    L1_delta = L2_delta.dot(W.T) * disigmoid(L1)
    W_C = lr * L1.T.dot(L2_delta)
    V_C = lr * X.T.dot(L1_delta)
    W = W + W_C
    V = V + V_C


for i in range(50000):
    update()
    if i % 500 == 0:
        L1 = sigmoid(np.dot(X, V))
        L2 = sigmoid(np.dot(L1, W))
        print('Error:', np.mean(np.abs(Y.T - L2)))

L1 = sigmoid(np.dot(X, V))
L2 = sigmoid(np.dot(L1, W))
print(L2)

def judge(x):
    if x>=0.5:
        return 1
    else:
        return 0

for i in map(judge,L2):
    print(i)


