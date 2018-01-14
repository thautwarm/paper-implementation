# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 16:47:22 2018

@author: misakawa
"""

import matplotlib
import platform
if 'windows' in platform.architecture()[1].lower():
    pass
else:
    matplotlib.use("Agg")
from definition import *
import os
from linq import Flow
from linq.standard.general import Map
from skimage import data, img_as_float
from pipe_fn import infix, and_then
from matplotlib import pyplot as plt
from math import exp
from itertools import cycle
import dill
from noise_maker import poisson_noise, gaussian_noise

model = torch.load('model_denoise', pickle_module=dill)
print('load model')

train_dir = './rainy_image_dataset/ground truth'
raw_sources = Flow(os.listdir(train_dir))
train_data_size = 500
test_data_size = 100
epochs = 100
lr = 0.01
batch_group_num = 5
loss_fn = torch.nn.MSELoss(size_average=True)

def mixed_noise(imgs_flow: np.ndarray):
    return and_then(
                gaussian_noise,  # 加高斯噪声
                poisson_noise)(imgs_flow)  # 浮点数张量 [0, 255]->[0, 1]

def DataIOStream(raw_src: Flow):
    return (raw_src
            .Filter(lambda x: x.endswith('.jpg'))  # select jpg files/选取jpg格式文件
            .Map(lambda x: os.path.join(train_dir, x))  # 拿到ground truth数据
            .Map(data.imread)
            .Map(lambda im:[im, 
                            mixed_noise(im), 
                            gaussian_noise(im), 
                            poisson_noise(im)] | infix/Map@img_as_float)
            .Map(to_batch))

test_batches = DataIOStream(raw_sources
                                .Drop(train_data_size)
                                .Take(test_data_size))
    
for test in test_batches.Unboxed():
    test_samples, test_targets = test    
    details, test_samples, test_targets = data_preprocessing(test_samples, test_targets)
    prediction = model(details, test_samples)
    
    pic = prediction.data.numpy()[0].clip(0, 1)
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