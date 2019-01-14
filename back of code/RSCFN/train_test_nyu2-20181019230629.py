# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-04-25 19:03:52
# @Last Modified by:   yulidong
# @Last Modified time: 2018-08-06 13:13:55
import scipy.io
import numpy as np
import os
data=scipy.io.loadmat('/home/lidong/Documents/datasets/nyu/nyu2_test_index.mat')
test1=data['testNdxs']-1
data=scipy.io.loadmat('/home/lidong/Documents/datasets/nyu/scenes.mat')
data=data['scenes']
scenes=[]
test_scenes=[]
train_scenes=[]
for i in range(len(data)):
    scenes.append(data[i][0][0])
for i in range(len(test1)):
    if not scenes[test1[i][0]] in test_scenes:
        test_scenes.append(scenes[test1[i][0]])
for i in range(len(scenes)):
    if not scenes[i] in test_scenes and not scenes[i] in train_scenes:
        train_scenes.append(scenes[i])
train_scenes.sort()
test_scenes.sort()
np.save('/home/lidong/Documents/datasets/nyu/train_scenes.npy',train_scenes)
print(len(test_scenes))
print(len(train_scenes))
print(len(test_scenes))
