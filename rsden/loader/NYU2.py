# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-04-25 23:06:40
# @Last Modified by:   yulidong
# @Last Modified time: 2019-01-29 16:18:06


import os
import torch
import numpy as np
import scipy.misc as m
import cv2
from torch.utils import data
from python_pfm import *
from rsden.utils import recursive_glob
import torchvision.transforms as transforms
import torchvision.transforms.functional as tf
import torch.nn.functional as F
import random
alpha=0.7132995128631592
#alpha=0.7
#alphat=0.3
beta=9.99547004699707
class NYU2(data.Dataset):
    def __init__(self, root, split="train", is_transform=True, img_size=(480,640),task='depth',index_bank=None):
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
            self.path=os.path.join('/home/dataset2/nyu/nyu2/train/')
            self.all_data=os.listdir('/home/dataset2/nyu/nyu2/train/')
            self.all_data.sort(key=lambda x:int(x[:-4]))
            if index_bank==None:
                self.files=np.arange(0,len(self.all_data))
            else:
                self.files=index_bank
            #self.files.sort(key=lambda x:int(x[:-4]))
            if len(self.files)<1:
                raise Exception("No files for %s found in %s" % (split, self.path))

            print("Found %d in %s images" % (len(self.files), self.path))

        if self.split=='test':
            self.path=os.path.join('/home/dataset2/nyu/nyu2/test/')
            self.files=os.listdir(self.path)
            self.all_data=os.listdir('/home/dataset2/nyu/nyu2/test/')
            self.all_data.sort(key=lambda x:int(x[:-4]))
            if index_bank==None:
                self.files=np.arange(0,len(self.all_data))
            else:
                self.files=index_bank
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
            self.r=7
            self.m=7
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
        # if self.split=='train':
        #     if index<self.l1:
        #         data=np.load(os.path.join(self.path,self.files[index]))
        #     elif index<self.l2:
        #         data=np.load(os.path.join('/home/dataset2/nyu/nyu1/train/',self.files[index]))
        #     else:
        #         data=np.load(os.path.join('/home/dataset2/nyu/nyu2/train/',self.files[index]))
        # else:
        #     data=np.load(os.path.join(self.path,self.files[index]))
        data=np.load(os.path.join(self.path,self.all_data[self.files[index]]))
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
        #print(torch.max(region))
        #exit()
        return img, depth,region,segments,image

    def transform(self, img, depth,region,segments):
        """transform

        :param img:
        :param depth:
        """
        img = img[:,:,:]
        img = img.astype(np.float32)/255
        #print(img.shape)

        # Resize scales images from 0 to 255, thus we need
        # to divide by 255.0
        #img = torch.from_numpy(img).float()
        depth = torch.from_numpy(depth).float().unsqueeze(0).unsqueeze(0)

        #print(d)
        segments=torch.from_numpy(segments).float().unsqueeze(0)

        region=torch.from_numpy(region).float().unsqueeze(0)
        #img = img.astype(float) / 255.0
        # NHWC -> NCHW
        #img = img.transpose(1,2,0)
        topil=transforms.ToPILImage()
        totensor=transforms.ToTensor()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        img=totensor(img)
        image=img.unsqueeze(0)+0
        #image=image/torch.max(image)

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
        # img=F.interpolate(img,scale_factor=1/2,mode='bilinear',align_corners=False).squeeze()[:,6:-6,8:-8]
        # image=F.interpolate(image,scale_factor=1/2,mode='bilinear',align_corners=False).squeeze()[:,6:-6,8:-8]
        # #print(depth.shape)
        # depth=F.interpolate(depth,scale_factor=1/2,mode='bilinear',align_corners=False).squeeze()[6:-6,8:-8]
        # region=F.interpolate(region,scale_factor=1/2,mode='bilinear',align_corners=False).squeeze()[6:-6,8:-8]
        # segments=F.interpolate(segments,scale_factor=1/2,mode='bilinear',align_corners=False).squeeze()[6:-6,8:-8]
        if self.split=='train':
            one=torch.ones(1)
            scale=random.uniform(1, 1.3)
            mask=(depth>alpha)&(depth<beta)
            mask=mask.float()
            #print(torch.sum(mask))
            h=int(240*scale)
            w=int(320*scale)
            md=torch.max(depth)
            #print(md)
            mr=torch.max(region)
            ms=torch.max(segments)
            #print(torch.max(image))
            img=tf.resize(topil(img.squeeze(0)),[h,w])
            image=tf.resize(topil(image.squeeze(0)),[h,w])
            #print(torch.max(totensor(image)))
            depth = tf.resize(topil(depth.squeeze(0)/md),[h,w])
            mask = tf.resize(topil(mask.squeeze(0)),[h,w])
            #print(segments.shape)
            segments=tf.resize(topil(segments.squeeze(0)/ms),[h,w])
            region=tf.resize(topil(region.squeeze(0)/mr),[h,w])
            i,j,h,w=transforms.RandomCrop.get_params(img,output_size=[228,304])
            r=random.uniform(-5, 5)
            sigma=random.uniform(0, 0.04)
            img=tf.rotate(img,r)
            image=tf.rotate(image,r)
            depth=tf.rotate(depth,r)
            mask=tf.rotate(mask,r)
            segments=tf.rotate(segments,r)
            region=tf.rotate(region,r)

            img=tf.crop(img,i,j,h,w)
            image=tf.crop(image,i,j,h,w)
            depth=tf.crop(depth,i,j,h,w)
            mask=tf.crop(mask,i,j,h,w)
            segments=tf.crop(segments,i,j,h,w)
            region=tf.crop(region,i,j,h,w)
            if random.uniform(0, 1)>0.5:
                img=tf.hflip(img)
                image=tf.hflip(image)
                depth=tf.hflip(depth)
                mask=tf.hflip(mask)
                segments=tf.hflip(segments)
                region=tf.hflip(region)
            # brightness=random.uniform(0, 0.4)
            # contrast=random.uniform(0, 0.4)
            # saturation=random.uniform(0, 0.4)
            # hue=random.uniform(0, 0.2)
            color=transforms.ColorJitter(0.4,0.4,0.4,0.4)
            img=color(img)
            gamma=random.uniform(0.7, 1.5)
            img=tf.adjust_gamma(img,gamma)
            img=totensor(img)
            r=random.uniform(0.8, 1.2)
            g=random.uniform(0.8, 1.2)
            b=random.uniform(0.8, 1.2)
            img[:,:,0]*=r
            img[:,:,1]*=g
            img[:,:,2]*=b
            gaussian=torch.zeros_like(img).normal_()*sigma
            img=img+gaussian
            img=img.clamp(min=0,max=1)
            image=img+0
            img=normalize(img)

            #image=totensor(image)
            #print(torch.max(image))
            #print(torch.max(depth),scale)
            depth=totensor(depth)*md/scale
            mask=totensor(mask)
            #print(torch.sum(mask))
            depth=torch.where(mask>0,depth,torch.zeros(1).float())
            #print(torch.max(depth))
            #print(torch.max(depth),scale)
            region=totensor(region)*mr
            segments=totensor(segments)*ms
            depth=torch.where(depth>beta,beta*one,depth)
            depth=torch.where(depth<alpha,alpha*one,depth)
            #exit()

        else:
            one=torch.ones(1)
            mask=(depth>alpha)&(depth<beta)
            mask=mask.float()
            img=img.unsqueeze(0)
            img=F.interpolate(img,scale_factor=1/2,mode='bilinear',align_corners=False).squeeze()
            image=F.interpolate(image,scale_factor=1/2,mode='bilinear',align_corners=False).squeeze()
            #print(depth.shape)
            #depth=F.interpolate(depth,scale_factor=1/2,mode='bilinear',align_corners=False).squeeze()
            #mask=F.interpolate(mask,scale_factor=1/2,mode='bilinear',align_corners=False).squeeze()
            #print(torch.sum(mask))
            #depth=torch.where(mask>=1,depth,torch.zeros(1).float())
            depth=torch.where(depth>beta,beta*one,depth)
            depth=torch.where(depth<alpha,alpha*one,depth)
            #depth=depth.squeeze()
            region=F.interpolate(region,scale_factor=1/2,mode='nearest').squeeze()

            segments=F.interpolate(segments,scale_factor=1/2,mode='nearest').squeeze()
            image=img[...,6:-6,8:-8]+0
            img=normalize(img)[...,6:-6,8:-8]
            segments=segments[...,6:-6,8:-8]+0
            region=region[...,6:-6,8:-8]+0
        #exit()
        # img=img.squeeze()
        # #print(depth.shape)
        # depth=depth.squeeze()
        # region=region.squeeze()
        # segments=segments.squeeze()
        #print(img.shape,image.shape,region.shape,segments.shape)
        return img, depth,region,segments,image
