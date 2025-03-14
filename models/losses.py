import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.utils import save_image
def save_images(tensor_data, output_folder, num_examples=4):
    """
    保存影像数据到指定文件夹，并可视化一些影像示例。

    Parameters:
    - tensor_data: torch.Tensor, 形状为 (num_samples, num_channels, height, width) 的影像数据
    - output_folder: str, 输出文件夹的路径
    - num_examples: int, 可视化的影像示例数量，默认为 4
    """
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 循环保存每一帧影像
    for i in range(tensor_data.size(0)):
        for j in range(tensor_data.size(1)):
            # 获取单帧影像数据
            single_image = tensor_data[i, j]

            # 将Tensor转换为NumPy数组，然后转换为PIL Image对象
            pil_image = transforms.ToPILImage()(single_image)

            # 构建文件名
            filename = os.path.join(output_folder, f'image_{i}_channel_{j}.png')

            # 保存影像
            pil_image.save(filename)

    print(f'All images have been saved to {output_folder}.')

    # 可视化一些影像数据示例
    fig, axs = plt.subplots(num_examples, 2, figsize=(10, 10))

    for k in range(num_examples):
        i = np.random.randint(tensor_data.size(0))
        j = np.random.randint(tensor_data.size(1))

        # 显示单帧影像数据
        image_np = tensor_data[i, j].detach().cpu().numpy()  # 转换为NumPy数组
        axs[k, 0].imshow(np.transpose(image_np, (1, 2, 0)), cmap='gray')  # 注意转置通道顺序
        axs[k, 0].set_title(f'Image {i}, Channel {j}')
        axs[k, 0].axis('off')

        # 如果需要，你也可以保存这些示例影像
        example_filename = os.path.join(output_folder, f'example_image_{i}_channel_{j}.png')
        save_image(tensor_data[i, j], example_filename)
        print(f'Saved example image {k} to {example_filename}')

    plt.tight_layout()
    plt.show()
def cross_entropy(input, target, weight=None, reduction='mean',ignore_index=255):
    """
    logSoftmax_with_loss
    ignore_index:指定被忽略的目标值的索引。如果目标值等于该索引，则不计算该样本的损失。
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    input:(8,2,256,256) target:(8,1,256,256)
    """
    target = target.long()
    #print(target.dim())#5
    #print(input.shape)
    #print(target.shape)
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if input.shape[-1] != target.shape[-1]:
        # print(input.shape[-1])#256
        # print(target.shape[-1])#3
        # print(input.shape)#torch.Size([8, 2, 256, 256]) N*C*H*W
        # print(target.shape)#torch.Size([8, 1, 256, 256, 3])N*1*H*W,/ N*H*W
        target.squeeze(4) #(8,256,256)
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)
    return F.cross_entropy(input=input, target=target, weight=weight,
                           ignore_index=ignore_index, reduction=reduction)


def Dice_loss_binary(input, target, reduction='mean',smooth=1e-8, p=1):
    """Dice loss of binary class
    Args:
    smooth: A float number to smooth loss, and avoid NaN error, default: 1
    p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
    predict: A tensor of shape [N, *]
    target: A tensor of shape same with predict
    reduction: Reduction method to apply, return mean over batch if 'mean',
        return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    num_classes = input.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[target.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(input)
        neg_prob = 1 - pos_prob
        input2 = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes,device=target.device)[target.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        # predict1 = input.cpu().detach().numpy()
        input2 = F.softmax(input, dim=1)  #(8,2,256,256)
        # predict2 = input2.cpu().detach().numpy()
    target2 = true_1_hot.type(input.type())
    dims = (0,) + tuple(range(2, target.ndimension()))
    num = torch.sum(input2 * target2, dims)
    den = torch.sum(input2.pow(p) + target2.pow(p), dims)
    dice_loss = 1-(2. * num/ (den + smooth))
    if reduction == 'mean':
        return dice_loss.mean()
    elif reduction == 'sum':
        return dice_loss.sum()
    elif reduction == 'none':
        return dice_loss
    else:
        raise Exception('Unexpected reduction {}'.format(reduction))
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss


class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        # self.kernel = torch.matmul(k.t(), k).unsqueeze(0).unsqueeze(0)
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(2, 1, 1, 1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)
    def true_hot(self,input,target):
        num_classes = input.shape[1]
        true_1_hot = torch.eye(num_classes, device=target.device)[target.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        input2 = F.softmax(input, dim=1)  # (8,2,256,256)
        target2 = true_1_hot.type(input.type())
        return input2,target2
    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)  # filter
        down = filtered[:, :, ::2, ::2]  # downsample
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4  # upsample
        filtered = self.conv_gauss(new_filter)  # filter
        diff = current - filtered
        return diff

    def forward_save_edge(self, x, y):
        # Ensure x and y are in binary range (0 or 1)
        num_classes = x.shape[1]
        true_1_hot = torch.eye(num_classes,device=y.device)[y.squeeze(0)]

        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        x = F.softmax(x, dim=1)  # (8,2,256,256)
        y = true_1_hot.type(x.type())

        x = torch.clamp(x, 0, 1)
        y = torch.clamp(y, 0, 1)

        # Compute loss
        x=self.laplacian_kernel(x)
        y=self.laplacian_kernel(y)
        return x,y
    def forward(self, x, y):
        # Ensure x and y are in binary range (0 or 1)
        x, y = self.true_hot(x, y)
        x = torch.clamp(x, 0, 1)#(8,2,256,256)
        y = torch.clamp(y, 0, 1)#(8,2,256,256)

        # Compute loss
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss

class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        self.CharbonnierLoss = CharbonnierLoss()
        self.EdgeLoss = EdgeLoss()

    def forward(self, x, y):
        loss1 = self.CharbonnierLoss(x, y)
        loss2 = self.EdgeLoss(x, y)
        lambda1 = loss1.data / (loss1.data + loss2.data)
        lambda2 = loss2.data / (loss1.data + loss2.data)
        loss = lambda1*loss1 + lambda2*loss2
        return loss

def new_loss(input, target,weight=1):
    #print('dice')
    loss_cross_entropy=cross_entropy(input, target)
    Dice_loss=Dice_loss_binary(input, target)
    return (loss_cross_entropy+weight*Dice_loss)
def entropy_Dice_edge_loss(input, target,weight=1):
    #print('dice')
    loss_cross_entropy=cross_entropy(input, target)
    Dice_loss=Dice_loss_binary(input, target)
    # Edge= CELoss()
    Edge = EdgeLoss()
    Edge_loss=Edge(input, target)
    return (loss_cross_entropy+weight*Dice_loss+10*Edge_loss)
def edge_loss(input, target):
    loss_cross_entropy=cross_entropy(input, target)
    Dice_loss=Dice_loss_binary(input, target)
    loss_binary=loss_cross_entropy+Dice_loss
    Edge = EdgeLoss()
    Edge_loss=Edge(input, target)
    # lambda1 = loss_binary.data / (loss_binary.data + Edge_loss.data)
    # lambda2 = Edge_loss.data / (loss_binary.data + Edge_loss.data)
    # loss = lambda1 * loss_binary + lambda2 * Edge_loss
    loss = loss_binary + Edge_loss
    return loss

