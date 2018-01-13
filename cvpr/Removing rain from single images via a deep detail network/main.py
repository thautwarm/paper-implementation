from definition import *
import os
from linq import Flow
from skimage import data, transform
from matplotlib import pyplot as plt

model = RainRemoval(3)

# 数据下载地址见: http://smartdsp.xmu.edu.cn/cvpr2017.html
# download data-sets here: http://smartdsp.xmu.edu.cn/cvpr2017.html
train_dir = './rainy_image_dataset/ground truth'
test_dir = './rainy_image_dataset/rainy image'
pos = 25
neg = 5
epochs = 15
lr = 0.1
loss_fn = torch.nn.MSELoss()


images = (
    Flow(os.listdir(train_dir))
        .Filter(lambda x: x.endswith('.jpg'))  # select jpg files/选取jpg格式文件
        .Map(lambda x: [os.path.join(train_dir, x)] +
                       [os.path.join(test_dir, x[:-4] + "_" + str(i) + '.jpg')
                        for i in range(1, 15)])  # 将训练集和数据集地址合并
        .Map(lambda img_file_names: list(map(and_then(data.imread,  # 读取图像
                                                      lambda x: transform.rescale(x, 0.5)*255,
                                                      lambda x: x.astype(float)),  # 浮点数张量
                                             img_file_names)))
        .Take(pos + neg)  # 选取前pos + neg个batch， 每个batch有28个图片，其中有14个是相同的target picture
        .ToList()
        .Unboxed()
)

batches = [(np.stack(samples),  # X
            np.stack([target] * len(samples)))  # y
           for target, *samples in images]

train_batches = batches[:pos]
test_batches = batches[-neg:]

loss = None

for epoch in range(epochs):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for train in train_batches:
        opt.zero_grad()
    
        train_samples, train_targets = train
        
        # 内存不足，只能取少一点的数据
        train_samples = train_samples[:5]
        train_targets = train_targets[:5]

        details, train_samples, train_targets = data_preprocessing(train_samples, train_targets)

        prediction = model(details, train_samples)

        loss = loss_fn(prediction, train_targets)
        print('epoch ' + str(epoch) + ': ', loss.data.numpy())
        loss.backward()
        opt.step()
        
        if loss is not None and loss.data.numpy()[0] > 400: 
            for _ in range(5):
            
                opt.zero_grad()
        
                train_samples, train_targets = train
                
                # 内存不足，只能取少一点的数据
                train_samples = train_samples[:5]
                train_targets = train_targets[:5]
        
                details, train_samples, train_targets = data_preprocessing(train_samples, train_targets)
        
                prediction = model(details, train_samples)
        
                loss = loss_fn(prediction, train_targets)
                print('sub-epoch ' + str(_) + ': ', loss.data.numpy())
                loss.backward()
        
                opt.step()

    lr *= 0.1

for test in test_batches:
    test_samples, test_targets = test    
    details, test_samples, test_targets = data_preprocessing(test_samples, test_targets)
    prediction = model(details, test_samples)
    
    pic = prediction.data.numpy().clip(0, 255)[0]
    plt.figure()
    plt.title('raw')
    plt.imshow(pic.transpose(1, 2, 0).astype(np.uint8))
    
    pic = prediction.data.numpy().clip(0, 255)[0]
    plt.figure()
    plt.title('prediction')
    plt.imshow(pic.transpose(1, 2, 0).astype(np.uint8))
    
    pic = test_samples.data.numpy().clip(0, 255)[0]
    plt.figure()
    plt.title('prediction')
    plt.imshow(pic.transpose(1, 2, 0).astype(np.uint8))
    
    plt.show()
    
    
    
