# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 16:53:51 2018

@author: misakawa
"""

from linq.standard.general import Map
from pipe_fn import infix, and_then
import numpy as np

poisson_noise = and_then(
    lambda x: x.astype(float),  # 转化为浮点数组
    infix/np.transpose@(2, 0, 1), # 将图片的(m x n x 3)转为(3 x m x n)
    infix/Map@(lambda x: x + np.random.poisson(20, x.shape[:2]).astype(float)), # 每个色道都泊松噪声化
    infix/np.stack@2, # 将三个转化后的色道在第3个维度组合起来，形成(m x n x 3)规模的张量。
    lambda x: x.clip(0, 255).astype(np.uint8))

gaussian_noise = and_then(
    lambda x: x.astype(float),  # 转化为浮点数组
    infix/np.transpose@(2, 0, 1), # 将图片的(m x n x 3)转为(3 x m x n)
    infix/Map@(lambda x: x + np.random.normal(0, 6.0, size=(x.shape[:2]))), # 每个色道都高斯噪声化
    infix/np.stack@2, # 将三个转化后的色道在第3个维度组合起来，形成(m x n x 3)规模的张量。
    lambda x: x.clip(0, 255).astype(np.uint8))