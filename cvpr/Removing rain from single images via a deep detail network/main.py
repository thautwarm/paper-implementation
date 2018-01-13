from definition import *
import os
from linq import Flow
from skimage import data

model = RainRemoval(3)

train_dir = './rainy_image_dataset/ground truth'
test_dir = './rainy_image_dataset/rainy image'
pos = 15
neg = 5
epochs = 60
lr = 0.1
loss_fn = torch.nn.MSELoss()

images = (
    Flow(os.listdir(train_dir))
        .Filter(lambda x: x.endswith('.jpg'))
        .Map(lambda x: [os.path.join(train_dir, x)] +
                       [os.path.join(test_dir, x[:-4] + "_" + str(i) + '.jpg') for i in range(1, 15)])
        .Map(lambda img_file_names: list(map(and_then(data.imread,
                                                      lambda x: x / 255),
                                             img_file_names)))
        .Take(pos + neg)
        .ToList()
        .Unboxed()
)

batches = [(np.stack(samples),  # X
            np.stack([target] * len(samples)))  # y
           for target, *samples in images]

train_batches = batches[:pos]
test_batches = batches[-neg:]

for _ in range(epochs):
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    for train in train_batches:
        opt.zero_grad()

        train_samples, train_targets = train
        train_samples = train_samples[:5]
        train_targets = train_targets[:5]

        details, train_samples, train_targets = data_preprocessing(train_samples, train_targets)

        prediction = model(details, train_samples)

        loss = loss_fn(prediction, train_targets)
        print('epoch ' + str(_) + ': ', loss.data.numpy())
        loss.backward()

        opt.step()

    lr *= 0.1
