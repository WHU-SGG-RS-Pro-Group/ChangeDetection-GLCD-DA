import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import cv2
import torch
from PIL import Image
import numpy as np
from torchvision.transforms.functional import rgb_to_grayscale
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def conv_gauss(img, kernel):
    img = F.pad(img, (2, 2, 2, 2), mode='reflect')
    out = F.conv2d(img, kernel, groups=img.shape[1])
    return out

def downsample(x):
    return x[:, :, ::2, ::2]
def gauss_kernel(channels=3, cuda=True):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                            [4., 16., 24., 16., 4.],
                            [6., 24., 36., 24., 6.],
                            [4., 16., 24., 16., 4.],
                            [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    if cuda:
        kernel = kernel.cuda()
    return kernel
def upsample(x, channels):
    device = x.device
    cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
    cc = cc.permute(0, 1, 3, 2)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
    x_up = cc.permute(0, 1, 3, 2)
    return conv_gauss(x_up, 4 * gauss_kernel(channels).to(device))
def make_laplace_pyramid(img, level, channels):
    current = img
    pyr = []
    device = img.device
    for _ in range(level):
        filtered = conv_gauss(current, gauss_kernel(channels).to(device))
        down = downsample(filtered)
        up = upsample(down, channels)
        if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
            up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
        diff = current - up
        pyr.append(diff)
        current = down
    pyr.append(current)
    return pyr
if __name__ == '__main__':
    # 加载图像并转换为 RGB 格式的 PIL 图像
    img = cv2.imread('.\samples_CALIFORNIA\B_SAR/California_9_10_256_256.png', 1).astype(float)
    # img = np.transpose( img, (2, 0, 1))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.ToTensor()
    img = transform(img)

    img = img.to(device, dtype=torch.float)
    img=torch.unsqueeze(img, dim=0)

    grayscale_img_x1 = rgb_to_grayscale(img)
    edge_feature_x1 = make_laplace_pyramid(grayscale_img_x1, 5, 1)
    x=torch.rand(1,32,64,64).to(device)
    xsize = x.size()[2:]
    # high-frequency feature
    edge_input = F.interpolate(edge_feature_x1[1], size=xsize, mode='bilinear', align_corners=True)
    input_feature = x * edge_input

    # 保存和显示 edge_feature_x1 的内容
    output_folder = '.\California_middle\edge/'  # 指定保存文件夹路径
    # 假设 edge_feature_x1 是一个数组列表，显示每一层的图像
    for i, edge in enumerate(edge_feature_x1):

        edge = torch.squeeze(edge, dim=0)
        edge = torch.squeeze(edge, dim=0)
        edge=edge.cpu()
        # 保存图像
        # plt.figure(figsize=(6, 6))
        # plt.title(f'Edge Feature Level {i + 1}')
        # plt.imshow(edge, cmap='gray')
        # plt.axis('off')
        # plt.show()

