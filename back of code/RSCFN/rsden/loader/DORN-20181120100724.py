# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-04-25 23:06:40
# @Last Modified by:   yulidong
# @Last Modified time: 2018-11-20 00:11:31


import os
import torch
import numpy as np
import scipy.misc as m
import cv2
from torch.utils import data
from python_pfm import *
import torchvision.transforms as transforms
import torch.nn.functional as F
import random

path=os.path.join('/home/dataset/datasets/nyu2_depth/npy_data')
files=os.listdir(path)
alpha=100
beta=0
min=[]
max=[]
for i in range(len(files)):
    data=np.load(os.path.join(path,files[i]))
    depth = data[:,:,3]
    depth=np.where(depth==0,np.mean(depth),depth)
    depth=np.where(depth==10,np.mean(depth),depth)
    alpha=np.min([alpha,np.max([0,np.min(depth)])])
    beta=np.max([beta,np.max(depth)])
    min.append(np.max([0,np.min(depth)]))
    max.append(np.max(depth))
    print(i,alpha,beta,min[-1],max[-1])
print(alpha,beta)
#0.7132995128631592 9.99547004699707
#0.014277142867333174 9.999999202576088