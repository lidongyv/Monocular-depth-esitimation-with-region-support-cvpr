# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-04-25 23:06:40
# @Last Modified by:   yulidong
# @Last Modified time: 2018-10-31 17:01:57


import os
import torch
import numpy as np
import scipy.misc as m
import cv2
from torch.utils import data
from python_pfm import *
from rsden.utils import recursive_glob
import torchvision.transforms as transforms
import torch.nn.functional as F
import random
class NYU(data.Dataset):
    def __init__(self, root, split="train", is_transform=True, img_size=(480,640),task='depth'):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        """
        self.root = root
        self.split = split
        self.num=0
        self.is_transform = is_transform
        self.n_classes = 64  # 0 is reserved for "other"
        self.img_size = img_size if isinstance(img_size, tuple) else (480, 640)
        self.stats={'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}
        if self.split=='train':
            self.path=os.path.join(self.root)
        if self.split=='test':
            self.path=os.path.join('/home/dataset2/nyu/nyu2/test/')
        self.files=os.listdir(self.path)
        self.files.sort(key=lambda x:int(x[:-4]))
        if len(self.files)<1:
            raise Exception("No files for %s found in %s" % (split, self.path))

        print("Found %d in %s images" % (len(self.files), self.path))
        self.task=task
        if task=='depth':
            self.d=3
            self.r=5
        else:
            self.d=5
            self.r=7
        if task=='all':
            self.d=3
            self.r=7 
        if task=='visualize':
            self.d=3
            self.r=5 
        if task=='region':
            self.d=3
            self.r=3
            self.m=3
        self.length=self.__len__()
    def __len__(self):
        """__len__"""
        return len(self.files)

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        # index=index%3
        # print(index)

        data=np.load(os.path.join(self.path,self.files[index]))
        if self.task=='visualize':
            data=data[0,:,:,:]
        # if self.split=='train':
        #     h,w = data.shape[0],data.shape[1]
        #     th, tw = 240, 320
        #     x1 = random.randint(0, h - th)
        #     y1 = random.randint(0, w - tw)
        #     data=data[x1:x1+th,y1:y1+tw,:]
        #print(data.shape)
        img = data[:,:,0:3]
        # img=img[:,:,::-1]
        #dis=readPFM(disparity_path)
        #dis=np.array(dis[0], dtype=np.uint8)

        depth = data[:,:,self.d]
        #depth=np.load(os.path.join('/home/lidong/Documents/datasets/nyu/nyu2/all',self.files[index]))[:,:,3]
        region=data[:,:,self.r]
        region=np.reshape(region,[1,region.shape[0],region.shape[1]])
        segments = data[:,:,self.m]
        #segments=np.load(os.path.join('/home/lidong/Documents/datasets/nyu/nyu2/all',self.files[index]))[:,:,4]
        #print(segments.shape)
        segments=np.reshape(segments,[1,segments.shape[0],segments.shape[1]])
        if self.task=='visualize':
            rgb=img
            img, depth,region,segments = self.transform(img, depth,region,segments)
            return img, depth,segments,data

        if self.is_transform:
            img, depth,region,segments,image = self.transform(img, depth,region,segments)

        return img, depth,region,segments,image

    def transform(self, img, depth,region,segments):
        """transform

        :param img:
        :param depth:
        """
        img = img[:,:,:]
        #print(img.shape)
        img = img.astype(np.float32)/255
        # Resize scales images from 0 to 255, thus we need
        # to divide by 255.0
        #img = torch.from_numpy(img).float()
        depth = torch.from_numpy(depth).float().unsqueeze(0).unsqueeze(0)
        segments=torch.from_numpy(segments).float().unsqueeze(0)
        region=torch.from_numpy(region).float().unsqueeze(0)
        #img = img.astype(float) / 255.0
        # NHWC -> NCHW
        #img = img.transpose(1,2,0)
        totensor=transforms.ToTensor()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        img=totensor(img)
        image=img.unsqueeze(0)+0
        img=normalize(img).unsqueeze(0)
        #print(img.shape,depth.shape)
        #depth=depth[0,:,:]
        #depth = depth.astype(float)/32
        #depth = np.round(depth)
        #depth = m.imresize(depth, (self.img_size[0], self.img_size[1]), 'nearest', mode='F')
        #depth = depth.astype(int)
        #depth=np.reshape(depth,[1,depth.shape[0],depth.shape[1]])
        #classes = np.unique(depth)
        #print(classes)
        #depth = depth.transpose(2,0,1)
        #if not np.all(classes == np.unique(depth)):
        #    print("WARN: resizing segmentss yielded fewer classes")

        #if not np.all(classes < self.n_classes):
        #    raise ValueError("Segmentation map contained invalid class values")
        img=F.interpolate(img,scale_factor=1/2,mode='nearest').squeeze()
        image=F.interpolate(image,scale_factor=1/2,mode='nearest').squeeze()
        #print(depth.shape)
        depth=F.interpolate(depth,scale_factor=1/2,mode='nearest').squeeze()
        region=F.interpolate(region,scale_factor=1/2,mode='nearest').squeeze()
        segments=F.interpolate(segments,scale_factor=1/2,mode='nearest').squeeze()
        # img=img.squeeze()
        # #print(depth.shape)
        # depth=depth.squeeze()
        # region=region.squeeze()
        # segments=segments.squeeze()
        #print(region.shape,segments.shape)
        return img, depth,region,segments,image
