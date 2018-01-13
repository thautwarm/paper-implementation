import numpy as np
import torch
from torch.nn import Module, ReLU, Conv2d, BatchNorm2d
from torch import nn
from torch.autograd import Variable
from pipe_fn import and_then
import cv2


def val(x):
    return Variable(x, requires_grad=False)


def guided_filter(img, width=None, height=None, channel=3):
    """
    This codes comes from the author's team.
    这些是作者团队的代码。
    ... 所以这堆PEP8 warnings不能怪我:)
    ... PyCharm词汇量少，常见词汇的warnings不能怪我:)
    """
    r = 15
    if width is None or height is None:
        width, height = img.shape[:2]
    eps = 1.0
    batch_q = np.zeros((width, height, channel))

    for j in range(channel):
        """
        thautwarm的点评: 这里的I, p, mean_I, mean_p令人窒息
        """
        I = img[:, :, j]
        p = img[:, :, j]
        ones_array = np.ones([width, height])
        N = cv2.boxFilter(ones_array, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0)
        mean_I = cv2.boxFilter(I, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
        mean_p = cv2.boxFilter(p, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
        mean_Ip = cv2.boxFilter(I * p, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
        cov_Ip = mean_Ip - mean_I * mean_p
        mean_II = cv2.boxFilter(I * I, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
        var_I = mean_II - mean_I * mean_I
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I
        mean_a = cv2.boxFilter(a, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
        mean_b = cv2.boxFilter(b, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
        q = mean_a * I + mean_b
        batch_q[:, :, j] = q
    return batch_q


class SamePaddingConv2d(Module):
    """
    padding = 'SAME' in PyTorch
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias):
        super(SamePaddingConv2d, self).__init__()
        self.keep_shape = lambda x: nn.functional.pad(
            x, [0, kernel_size - 1, 0, kernel_size - 1])
        self.conv = Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias)
        self.relu = ReLU()
        self.batch_norm = BatchNorm2d(out_channels)

    def forward(self, x):
        return and_then(
            self.keep_shape,
            self.conv,
            self.relu,
            self.batch_norm)(x)


class RainRemoval(Module):
    """
    http://openaccess.thecvf.com/content_cvpr_2017/papers/Fu_Removing_Rain_From_CVPR_2017_paper.pdf
    """

    def __init__(self, group_num: int, kernel_size: int = 3):
        super(RainRemoval, self).__init__()

        self.conv_fst = SamePaddingConv2d(3, 16, kernel_size=kernel_size, bias=True)
        self.convs = nn.ModuleList()

        for idx in range(0, group_num, 2):
            self.convs.append(nn.ModuleList([
                SamePaddingConv2d(16, 16, kernel_size, True),
                SamePaddingConv2d(16, 16, kernel_size, True)
            ]))

        self.conv_end = SamePaddingConv2d(16, 3, kernel_size=kernel_size, bias=True)

    def forward(self, detail, img):
        last = self.conv_fst(detail)
        short_cut = last

        for conv in self.convs:
            for component in conv:
                last = component(last)
            last = last + short_cut
            short_cut = last

        return self.conv_end(last) + img


def data_preprocessing(train_samples, *accompanies):
    """
    1. 对图片做低通滤波得到base，使用原图减去base得到detail
    2. 将details, 原图，以及其他的图片(例如ground truth)转化到PyTorch对应的数据结构。
    """
    details = np.array(
        [train_sample - base
         for train_sample, base in
         zip(train_samples, map(guided_filter, train_samples))])

    # transpose the dimensions from m x n x 3 to 3 x m x n.
    details, train_samples, *accompanies = map(lambda x: x.transpose(0, 3, 1, 2),
                                               (details, train_samples, *accompanies))

    details, train_samples, *accompanies = map(
        # transform data type to make them compatible for PyTorch.
        and_then(
            torch.from_numpy,  # transform numpy.ndarray to torch.Tensor
            lambda x: x.float(),  # mark it as float type
            val,  # mark it as PyTorch's immutable variable.
        ),

        (details, train_samples, *accompanies))

    # pycharm fails in analyzing syntax here...
    return (details, train_samples, *accompanies)
