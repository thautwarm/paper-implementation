# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 18:08:59 2018

@author: misakawa
"""

from definition import *
import os
from linq import Flow
from skimage import data, img_as_float
from matplotlib import pyplot as plt
from math import exp
from itertools import cycle
import dill

model = torch.load('model', pickle_module=dill)
print('load model')

    
# 数据下载地址见: http://smartdsp.xmu.edu.cn/cvpr2017.html
# download data-sets here: http://smartdsp.xmu.edu.cn/cvpr2017.html
train_dir = './rainy_image_dataset/ground truth'
test_dir = './rainy_image_dataset/rainy image'
train_data_size = 500
test_data_size = 100
lr = 0.1
raw_sources = Flow(os.listdir(train_dir))

def DataIOStream(raw_src: Flow, num: int):
    return (raw_src
            .Take(num)
            .Then(cycle)
            .Filter(lambda x: x.endswith('.jpg'))  # select jpg files/选取jpg格式文件
            .Map(lambda x: [os.path.join(train_dir, x)] +
                           [os.path.join(test_dir, x[:-4] + "_" + str(i) + '.jpg')
                            for i in range(1, 3)])  # 将噪声数据和真实数据进行合并
            .Map(lambda img_file_names: list(map(and_then(data.imread,  # 读取图像
                                                          img_as_float),  # 浮点数张量 [0, 255]->[0, 1]
                                                 img_file_names)))
            .Map(to_batch))
    
test_batches = DataIOStream(raw_sources.Drop(train_data_size), test_data_size)
loss = None

for test in test_batches.Take(test_data_size).Unboxed():
    test_samples, test_targets = test    
    details, test_samples, test_targets = data_preprocessing(test_samples, test_targets)
    prediction = model(details, test_samples)
    
    pic = test_samples.data.numpy()[0].clip(0, 1)
    plt.figure()
    plt.title('raw')
    plt.imshow(pic.transpose(1, 2, 0))
    
    pic = prediction.data.numpy()[0].clip(0, 1)
    plt.figure()
    plt.title('prediction')
    plt.imshow(pic.transpose(1, 2, 0))
    
    pic = test_targets.data.numpy()[0].clip(0, 1)
    plt.figure()
    plt.title('target')
    plt.imshow(pic.transpose(1, 2, 0))
    
    plt.show()
    
    
    
