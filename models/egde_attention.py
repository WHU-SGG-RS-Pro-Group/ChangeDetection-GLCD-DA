import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import numpy as np
class edge_attention1(nn.Module):
    def __init__(self,inchannel = 1):
        super(edge_attention1,self).__init__()
        sobel_0 = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).astype(dtype='float32')
        sobel_0 = np.reshape(sobel_0, (1, 3, 3))
        sobel_45 = np.array([[-2.0, -1.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 1.0, 2.0]]).astype(dtype='float32')
        sobel_45 = np.reshape(sobel_45, (1, 3, 3))
        sobel_90 = np.array([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).astype(dtype='float32')
        sobel_90 = np.reshape(sobel_90, (1, 3, 3))
        sobel_135 = np.array([[0.0, -1.0, -2.0], [1.0, 0.0, -1.0], [2.0, 1.0, 0.0]]).astype(dtype='float32')
        sobel_135 = np.reshape(sobel_135, (1, 3, 3))
        sobel3d = np.concatenate(
            (np.repeat(sobel_0, inchannel, axis=0).reshape((1, inchannel, 3, 3)),
            np.repeat(sobel_45, inchannel, axis=0).reshape((1, inchannel, 3, 3)),
            np.repeat(sobel_90, inchannel, axis=0).reshape((1, inchannel, 3, 3)),
            np.repeat(sobel_135, inchannel, axis=0).reshape((1, inchannel, 3, 3))),
            axis = 0
        )
        self.conv1 = nn.Conv2d(in_channels = inchannel,out_channels = 4,kernel_size = 3, padding= 1,
                              stride = [1,1], bias = False)
        self.conv1.weight.data = torch.from_numpy(sobel3d)
        self.conv1.requires_grad = False
        self.sigmoid = nn.Sigmoid()
        self.conv2 = nn.Conv2d(4, 1, 1, padding=0, bias=False)
        # self.conv3 = nn.Conv2d(2, 1, 3, padding=1, bias=False)
        # self.conv1x1 = nn.Conv2d(64*2, 64, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        b,c,h,w = x.shape
        t = x.reshape(b*c,1,h,w)
        edge=self.conv1(t)
        edge = self.conv2(edge)
        # print("edge.shape",edge.shape)
        # sigmoid 1
        edge = edge.reshape(b, c, h, w)
        y = self.sigmoid(edge)
        out = y * x
        # sigmoid 2
        # y = self.sigmoid(edge)
        # y = y.resahpe(b, c, h, w)
        # x = y * x
        # edge = self.conv2(self.conv1(x))
        # y = self.sigmoid(edge)
        # x = y * x
        return out