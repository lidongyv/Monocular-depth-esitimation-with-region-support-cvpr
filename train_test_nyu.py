# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-04-25 19:03:52
# @Last Modified by:   yulidong
# @Last Modified time: 2018-10-09 14:59:51
import scipy.io
import numpy as np
import os
data=scipy.io.loadmat('/home/dataset2/nyu/nyu2/split_train_test.mat')
train=data['trainNdxs']
test=data['testNdxs']
# for i in range(len(train)):
#     filename=str(train[i][0]-1)+'.npy'
#     #print(os.path.join('home/lidong/Documents/datasets/nyu/train/',filename))
#     os.rename(os.path.join('/home/lidong/Documents/datasets/nyu/train/',filename),os.path.join('/home/lidong/Documents/datasets/nyu/test/',filename))
for i in range(len(test)):
    filename=str(test[i][0]-1)+'.npy'
    #print(filename)
    os.rename(os.path.join('/home/dataset2/nyu/nyu2/train/',filename),os.path.join('/home/dataset2/nyu/nyu2/test/',filename))
# import matplotlib.pyplot as plt    
# import numpy as np
# import h5py
# import os    
# data =  h5py.File('/home/lidong/Documents/datasets/nyu/nyu_depth_v2_labeled.mat')
# names=data['scenes']
# print(str(names[0,0]))

