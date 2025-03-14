"""
变化检测数据集
"""

import os

import torch
from PIL import Image
import numpy as np

from torch.utils import data

from datasets.data_utils import CDDataAugmentation


"""
CD data set with pixel-level labels；
├─image
├─image_post
├─label
└─list
"""
#Add to pre-transfer images
# IMG_FOLDER_NAME = "A_OPT"
# IMG_FOLDER_NAME2 = "A_SAR"
# IMG_POST_FOLDER_NAME = 'B_SAR'
# IMG_POST_FOLDER_NAME2 = 'B_OPT'
# LIST_FOLDER_NAME = 'list'
# ANNOT_FOLDER_NAME = "label"
#Don't add pre-transfer images
IMG_FOLDER_NAME = "A_SAR"
IMG_POST_FOLDER_NAME = 'B_SAR'
LIST_FOLDER_NAME = 'list'
ANNOT_FOLDER_NAME = "label"

IGNORE = 255

label_suffix='.png' # jpg for gan dataset, others : png

def load_img_name_list(dataset_path):
    img_name_list = np.loadtxt(dataset_path, dtype=np.str_)#加载文件
    if img_name_list.ndim == 2: #ndim返回的是数组的维度，返回的只有一个数，该数即表示数组的维度。
        return img_name_list[:, 0]  #表示对一个二维数组，取该二维数组第一维中的所有数据，
    return img_name_list


def load_image_label_list_from_npy(npy_path, img_name_list):
    cls_labels_dict = np.load(npy_path, allow_pickle=True).item()
    return [cls_labels_dict[img_name] for img_name in img_name_list]


def get_img_post_path(root_dir,img_name):
    return os.path.join(root_dir, IMG_POST_FOLDER_NAME, img_name)

def get_img_post_path2(root_dir,img_name):
    return os.path.join(root_dir, IMG_POST_FOLDER_NAME2, img_name)


def get_img_path(root_dir, img_name):
    return os.path.join(root_dir, IMG_FOLDER_NAME, img_name)

def get_img_path2(root_dir, img_name):
    return os.path.join(root_dir, IMG_FOLDER_NAME2, img_name)


def get_label_path(root_dir, img_name):
    return os.path.join(root_dir, ANNOT_FOLDER_NAME, img_name.replace('.jpg', label_suffix))


class ImageDataset(data.Dataset):
    """VOCdataloder"""
    def __init__(self, root_dir, split='train', img_size=256, is_train=True,to_tensor=True):
        super(ImageDataset, self).__init__()
        self.root_dir = root_dir
        self.img_size = img_size
        self.split = split  # train | train_aug | val
        self.list_path = self.root_dir + '/' + LIST_FOLDER_NAME + '/' + self.split + '.txt'
        self.img_name_list = load_img_name_list(self.list_path)
        self.A_size = len(self.img_name_list)  # get the size of dataset A

        self.to_tensor = to_tensor
        if is_train:
            self.augm = CDDataAugmentation(
                img_size=self.img_size,
                with_random_hflip=False,
                with_random_vflip=False,
                with_scale_random_crop=False,
                with_random_blur=False,
            )
            # self.augm = CDDataAugmentation(
            #     img_size=self.img_size
            # )
        else:
            self.augm = CDDataAugmentation(
                img_size=self.img_size
            )
    def __getitem__(self, index):
        name = self.img_name_list[index]
        A_path = get_img_path(self.root_dir, self.img_name_list[index % self.A_size])
        A_path2 = get_img_path2(self.root_dir, self.img_name_list[index % self.A_size])
        B_path = get_img_post_path(self.root_dir, self.img_name_list[index % self.A_size])
        B_path2 = get_img_post_path2(self.root_dir, self.img_name_list[index % self.A_size])
        img = np.asarray(Image.open(A_path).convert('RGB'))
        img2 = np.asarray(Image.open(A_path2).convert('RGB'))
        img_B = np.asarray(Image.open(B_path).convert('RGB'))
        img_B2 = np.asarray(Image.open(B_path2).convert('RGB'))
        [img, img_B], _ = self.augm.transform([img, img_B], [], to_tensor=self.to_tensor)
        #Gloucester数据集和California数据集
        [img,img2, img_B], _ = self.augm.transform([img,img2, img_B], [], to_tensor=self.to_tensor)
        img=torch.cat([img,img2],dim=0)
        #Gloucester2数据集
        # [img, img_B, img_B2], _ = self.augm.transform([img, img_B, img_B2], [], to_tensor=self.to_tensor)
        # img_B = torch.cat([img_B, img_B2], dim=0)
        return {'A': img, 'B': img_B, 'name': name}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.A_size


class CDDataset(ImageDataset):

    def __init__(self, root_dir, img_size, split='train', is_train=True, label_transform=None,
                 to_tensor=True):
        super(CDDataset, self).__init__(root_dir, img_size=img_size, split=split, is_train=is_train,
                                        to_tensor=to_tensor)
        self.label_transform = label_transform

    def __getitem__(self, index):
        name = self.img_name_list[index]
        A_path = get_img_path(self.root_dir, self.img_name_list[index % self.A_size])
        # A_path2 = get_img_path2(self.root_dir, self.img_name_list[index % self.A_size])
        B_path = get_img_post_path(self.root_dir, self.img_name_list[index % self.A_size])
        # B_path2 = get_img_post_path2(self.root_dir, self.img_name_list[index % self.A_size])
        img = np.asarray(Image.open(A_path).convert('RGB'))
        # img2 = np.asarray(Image.open(A_path2).convert('RGB'))
        img_B = np.asarray(Image.open(B_path).convert('RGB'))
        # img_B2 = np.asarray(Image.open(B_path2).convert('RGB'))
        L_path = get_label_path(self.root_dir, self.img_name_list[index % self.A_size])

        label = np.array(Image.open(L_path), dtype=np.uint8)
        #  二分类中，前景标注为255
        if self.label_transform == 'norm':
            label = label // 255

        [img, img_B], [label] = self.augm.transform([img, img_B], [label], to_tensor=self.to_tensor)

        # Gloucester dataset and California dataset
        # [img, img2, img_B], [label] = self.augm.transform([img, img2, img_B], [label], to_tensor=self.to_tensor)
        # img = torch.cat([img, img2], dim=0)
        # Gloucester2 dataset
        # [img, img_B, img_B2], [label] = self.augm.transform([img, img_B, img_B2], [label], to_tensor=self.to_tensor)
        # img_B = torch.cat([img_B, img_B2], dim=0)
        # print(label.max())
        return {'name': name, 'A': img, 'B': img_B, 'L': label}

