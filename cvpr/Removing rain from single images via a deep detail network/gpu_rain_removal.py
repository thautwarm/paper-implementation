import matplotlib
import platform
if 'windows' in platform.architecture()[1].lower():
    pass
else:
    matplotlib.use("Agg")

from definition import *
import os
from linq import Flow
from skimage import data, img_as_float
from matplotlib import pyplot as plt
from math import exp
from itertools import cycle
import dill
try:    
    model = torch.load('model', pickle_module=dill)
    print('load model')
except:    
    print('new_model')
    model = RainRemoval(10)

model.cuda()


# 数据下载地址见: http://smartdsp.xmu.edu.cn/cvpr2017.html
# download data-sets here: http://smartdsp.xmu.edu.cn/cvpr2017.html
train_dir = './rainy_image_dataset/ground truth'
test_dir = './rainy_image_dataset/rainy image'
train_data_size = 500
test_data_size = 100
epochs = 115
batch_group_num = 3 
lr = 0.01
loss_fn = torch.nn.MSELoss(size_average=False)


def to_batch(image):
    target, *samples = image
    return (np.stack(samples),  # X
            np.stack([target] * len(samples)))
    
raw_sources = Flow(os.listdir(train_dir))

def DataIOStream(raw_src: Flow, num: int):
    return (raw_src
            .Take(num)
            .Filter(lambda x: x.endswith('.jpg'))  # select jpg files/选取jpg格式文件
            .Map(lambda x: [os.path.join(train_dir, x)] +
                           [os.path.join(test_dir, x[:-4] + "_" + str(i) + '.jpg')
                            for i in range(1, 3)])  # 将噪声数据和真实数据进行合并
            .Map(lambda img_file_names: list(map(and_then(data.imread,  # 读取图像
                                                          img_as_float),  # 浮点数张量 [0, 255]->[0, 1]
                                                 img_file_names)))
            .Map(to_batch))

train_batches = DataIOStream(raw_sources, train_data_size).ToList().Then(cycle)
print('data_loaded')

loss = None
try:
    for epoch in range(epochs):
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        Loss: list = []
        train_loss = None
        
        for inner_idx, train in enumerate(train_batches.Take(train_data_size).Unboxed()):
            
            opt.zero_grad()
        
            train_samples, train_targets = train
    
            details, train_samples, train_targets = data_preprocessing(train_samples, train_targets)
            
            
            prediction = model(details.cuda(), train_samples.cuda())
    
            loss = loss_fn(prediction, train_targets.cuda())
            if train_loss is not None:
                train_loss += loss
            else:
                train_loss = loss
            
            if inner_idx is not 0 and inner_idx % batch_group_num == 0:
                train_loss.backward()
                opt.step()
                print('current-minibatch-loss:', train_loss.cpu().data.numpy()[0])
                train_loss = None
            
            loss = loss.cpu().data.numpy()[0]
            Loss.append(loss)
        
        if inner_idx % batch_group_num is not 0:
            train_loss.backward()
            opt.step()
            print('current-minibatch-loss:', train_loss.cpu().data.numpy()[0])
            train_loss = None
            
        Loss: float = np.mean(Loss)
        print('epoch {}. lr {}. loss: {}'.format(epoch, lr, Loss))
finally:
    print('saving model')
    torch.save(model.cpu(), 'model', pickle_module=dill)
    
