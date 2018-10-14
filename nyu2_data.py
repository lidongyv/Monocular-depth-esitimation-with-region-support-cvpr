# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-04-25 18:05:58
# @Last Modified by:   yulidong
# @Last Modified time: 2018-07-27 22:38:09
import matplotlib.pyplot as plt    
import numpy as np
import h5py
import os    
data =  h5py.File('/home/lidong/Documents/datasets/nyu/nyu_depth_v2_labeled.mat')
# files=os.listdir('/home/lidong/Documents/datasets/nyu/train')
# start=len(files)
# files.sort(key=lambda x:int(x[:-4]))
#start=start-depths.shape[0]
keys=[]
values=[]
#shapes=[]
for k, v in data.items():
    keys.append(k)
    values.append(v)
    print(v)
depths=data['depths']
images=data['images']
labels=data['labels']


for i in range(depths.shape[0]):

    print(i)
    group=[]
    image=images[i,:,:,:].astype('float')
    image=np.transpose(image,[2,1,0])
    depth=depths[i,:,:].astype('float')
    depth=np.transpose(depth,[1,0])
    depth=np.reshape(depth,[depth.shape[0],depth.shape[1],1])
    label=labels[i,:,:].astype('float')
    label=np.transpose(label,[1,0])
    label=np.reshape(label,[label.shape[0],label.shape[1],1])
    group=np.concatenate((image,depth),2)
    group=np.concatenate((group,label),2)
    #np.save('/home/lidong/Documents/datasets/nyu/train/'+str(i+start)+'.npy',group)
    np.save('/home/lidong/Documents/datasets/nyu/nyu2/all/'+str(i)+'.npy',group)

