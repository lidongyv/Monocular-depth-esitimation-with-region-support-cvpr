# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-03-19 13:33:07
# @Last Modified by:   yulidong
# @Last Modified time: 2018-04-07 15:14:04

import os
import torch
import numpy as np
import scipy.misc as m
import cv2
from torch.utils import data
from python_pfm import *
from rsden.utils import recursive_glob


class SceneFlow(data.Dataset):


    def __init__(self, root, split="train", is_transform=True, img_size=(540,960)):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        """
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.n_classes = 9  # 0 is reserved for "other"
        self.img_size = img_size if isinstance(img_size, tuple) else (540, 960)
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.files = {}

        self.li = os.path.join(self.root,self.split,'li')
        self.ld = os.path.join(self.root,self.split,'ld')
        self.lr = os.path.join(self.root,self.split,'lr')

        self.files['li'] = recursive_glob(rootdir=self.li, suffix='.png')
        self.files['ld'] = recursive_glob(rootdir=self.ld, suffix='.pfm')
        self.files['lr'] = recursive_glob(rootdir=self.lr, suffix='.png')
        if not self.files['ld']:
            raise Exception("No files for ld=[%s] found in %s" % (split, self.ld))

        print("Found %d %s images" % (len(self.files['ld']), split))

    def __len__(self):
        """__len__"""
        return len(self.files['ld'])

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        img_path = self.files['li'][index].rstrip()
        disparity_path = self.files['ld'][index].rstrip()
        region_path=self.files['lr'][index].rstrip()
        img = cv2.imread(img_path)
        img = np.array(img, dtype=np.uint8)
        #dis=readPFM(disparity_path)
        #dis=np.array(dis[0], dtype=np.uint8)
        region = cv2.imread(region_path)
        region = np.array(region, dtype=np.uint8)

        if self.is_transform:
            img, region = self.transform(img, region)

        return img, region

    def transform(self, img, region):
        """transform

        :param img:
        :param region:
        """
        img = img[:, :,:]
        img = img.astype(np.float64)
        # Resize scales images from 0 to 255, thus we need
        # to divide by 255.0
        img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        region=region[:,:,0]
        region = region.astype(float)/32
        region = np.round(region)
        #region = m.imresize(region, (self.img_size[0], self.img_size[1]), 'nearest', mode='F')
        region = region.astype(int)
        region=np.reshape(region,[1,region.shape[0],region.shape[1]])
        classes = np.unique(region)
        #print(classes)
        #region = region.transpose(2,0,1)
        if not np.all(classes == np.unique(region)):
            print("WARN: resizing labels yielded fewer classes")

        if not np.all(classes < self.n_classes):
            raise ValueError("Segmentation map contained invalid class values")

        img = torch.from_numpy(img).float()
        region = torch.from_numpy(region).long()

        return img, region
