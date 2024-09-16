"""
file - dataset.py
Customized dataset class to loop through the AVA dataset and apply needed image augmentations for training.

Copyright (C) Yunxiao Shi 2017 - 2021
NIMA is released under the MIT license. See LICENSE for the fill license text.
"""

import os

import pandas as pd
from PIL import Image

import torch
from torch.utils import data
import torchvision.transforms as transforms
import torch.nn.functional as F

compare_fastsam = False

mean = [0.485, 0.456, 0.406]  # RGB
std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224), antialias=True),
    transforms.RandomCrop(448),
    transforms.RandomHorizontalFlip(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((512, 512), antialias=True),
    transforms.RandomCrop(448),

    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])



def dis_2_score(dis: torch.Tensor):
    """
    convert distance to score
    :param dis: distance
    :return: score, tensor
    """
    w = torch.linspace(1, 10, 10).to(dis.device)
    w_batch = w.repeat(dis.shape[0], 1)
    score = (dis * w_batch).sum(dim=1)
    return score


class AVADataset(data.Dataset):
    """AVA dataset

    Args:
        csv_file: a 11-column csv_file, column one contains the names of image files, column 2-11 contains the empiricial distributions of ratings
        root_dir: directory to the images
        transform: preprocessing and augmentation of the training images
    """  
    def __init__(self, args,train=True):
        csv_file = os.path.join(args.dataset.label_dir, 'train_labels.csv' if train else 'val_labels.csv')
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = args.dataset.data_dir
        self.device = args.train.device
        self.imgsz = args.dataset.imgsz
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.imgsz, self.imgsz), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.imgsz, self.imgsz), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform = train_transform if train else val_transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(self.annotations.iloc[idx, 0]) + '.jpg')
        image = Image.open(img_name).convert('RGB')
        # resize img
        image = image.resize((self.imgsz, self.imgsz))
        annotations = self.annotations.iloc[idx, 1:11].to_numpy()
        annotations = annotations.astype('float').reshape(-1, )
        sample = {'img_id': img_name, 'image': image, 'label': annotations}
        sample['score'] = dis_2_score(torch.tensor(sample['label']))

        if self.transform:
            sample['image'] = self.transform(sample['image'])
            # print(sample['image'].shape)

        return sample


class AVADatasetMP(data.Dataset):
    """AVA dataset

    Args:
        csv_file: a 11-column csv_file, column one contains the names of image files, column 2-11 contains the empiricial distributions of ratings
        root_dir: directory to the images
        transform: preprocessing and augmentation of the training images
    """

    def __init__(self, csv_file, root_dir, patch_num=5, imgsz=512, device='cpu', train=True):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.device = device
        self.imgsz = imgsz
        self.patch_num = patch_num
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        train_transform = transforms.Compose([
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
        ])
        val_transform = transforms.Compose([
            transforms.RandomCrop(224),
        ])
        self.transform = train_transform if train else val_transform

    def __len__(self):
        return len(self.annotations)

    def resize_image_keep_ratio(self, img, height, width, target_short_side=224):
        # 打开图像

        if height > width:
            ratio = target_short_side / width
            new_width = target_short_side
            new_height = int(height * ratio)
        else:
            ratio = target_short_side / height
            new_height = target_short_side
            new_width = int(width * ratio)

        # 按比例调整图像大小
        resized_img = F.interpolate(img.unsqueeze(0), size=(new_height, new_width), mode='bilinear', align_corners=False).squeeze(0)

        return resized_img

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(self.annotations.iloc[idx, 0]) + '.jpg')
        image = Image.open(img_name).convert('RGB')
        image = self.base_transform(image)
        height, width = image.shape[1], image.shape[2]
        if width < self.imgsz or height < self.imgsz:
            image = self.resize_image_keep_ratio(image, height, width, self.imgsz)

        # resize img
        patches = []
        for i in range(self.patch_num):
            patches.append(self.transform(image))
        image = torch.stack(patches, dim=0)
        annotations = self.annotations.iloc[idx, 1:11].to_numpy()
        annotations = annotations.astype('float').reshape(-1, )
        sample = {'img_id': img_name, 'image': image, 'label': annotations}
        sample['score'] = dis_2_score(torch.tensor(sample['label']))

        return sample


class NimaAVADataset(data.Dataset):
    """AVA dataset

    Args:
        csv_file: a 11-column csv_file, column one contains the names of image files, column 2-11 contains the empiricial distributions of ratings
        root_dir: directory to the images
        transform: preprocessing and augmentation of the training images
    """

    def __init__(self, csv_file, root_dir, imgsz=512, device='cpu', train=True):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.device = device
        self.imgsz = imgsz

        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((imgsz, imgsz), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((imgsz, imgsz), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform = train_transform if train else val_transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(self.annotations.iloc[idx, 0]) + '.jpg')
        image = Image.open(img_name).convert('RGB')
        # resize img
        image = image.resize((self.imgsz, self.imgsz))
        annotations = self.annotations.iloc[idx, 1:11].to_numpy()
        annotations = annotations.astype('float').reshape(-1, )
        sample = {'img_id': img_name, 'image': image, 'label': annotations}
        sample['score'] = dis_2_score(torch.tensor(sample['label']))

        if self.transform:
            sample['image'] = self.transform(sample['image'])
            # print(sample['image'].shape)

        return sample


if __name__ == "__main__":
    train_dataset = AVADatasetMP(csv_file='D:\\Dataset\\AVA\\labels\\train_labels.csv',
                                 root_dir='D:\\Dataset\\AVA\\images', imgsz=224, patch_num=5)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=10, shuffle=True, num_workers=0)

    for i, data in enumerate(train_loader):
        print(data['image'].shape)
        print(data['score'].shape)
        break
    print("Done!")
