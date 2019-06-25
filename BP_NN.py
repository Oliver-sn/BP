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
X = np.array([[1, 0, 1],
              [1, 0, 1],
              [1, 1, 0],
              [1, 1, 1]])
# 标签
Y = np.array([-1, 1, 1, -1])
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
