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
lr = 0.1
loss_fn = torch.nn.MSELoss(size_average=False)


def to_batch(image):
    target, *samples = image
    return (np.stack(samples),  # X
            np.stack([target] * len(samples)))
    
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
            .Map(to_batch)
        )

train_batches = DataIOStream(raw_sources, train_data_size)
#test_batches = DataIOStream(raw_sources.Drop(train_data_size), test_data_size)

loss = None
try:
    for epoch in range(epochs):
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        Loss: list = []
        for train in train_batches.Take(train_data_size).Unboxed():
            opt.zero_grad()
        
            train_samples, train_targets = train
            
            # 内存不足，只能取少一点的数据
    
            details, train_samples, train_targets = data_preprocessing(train_samples, train_targets)
            
            
            prediction = model(details.cuda(), train_samples.cuda())
    
            loss = loss_fn(prediction, train_targets.cuda())
            loss.backward()
            opt.step()
            
            loss = loss.cpu().data.numpy()[0]
            Loss.append(loss)
            print('after minibatch: loss =', Loss[-1])
    
        Loss: float = np.mean(Loss)
        print('epoch {}. lr {}. loss: {}'.format(epoch, lr, Loss))
        lr = 0.1 if Loss > 100 else 0.01
finally:
    torch.save(model.cpu(), 'model', pickle_module=dill)
        

#for test in test_batches.Take(test_data_size).Unboxed():
#    test_samples, test_targets = test    
#    details, test_samples, test_targets = data_preprocessing(test_samples, test_targets)
#    prediction = model(details, test_samples)
#    
#    pic = prediction.data.numpy()[0].clip(0, 1)
#    plt.figure()
#    plt.title('raw')
#    plt.imshow(pic.transpose(1, 2, 0))
#    
#    pic = prediction.data.numpy()[0].clip(0, 1)
#    plt.figure()
#    plt.title('prediction')
#    plt.imshow(pic.transpose(1, 2, 0))
#    
#    pic = test_targets.data.numpy()[0].clip(0, 1)
#    plt.figure()
#    plt.title('target')
#    plt.imshow(pic.transpose(1, 2, 0))
#    
#    plt.show()
    
    
    
