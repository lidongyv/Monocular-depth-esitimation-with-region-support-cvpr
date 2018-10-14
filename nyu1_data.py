# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-04-25 16:31:59
# @Last Modified by:   yulidong
# @Last Modified time: 2018-10-10 22:36:05
import matplotlib.pyplot as plt    
import numpy as np
import h5py    
data =  h5py.File('/home/dataset2/nyu/nyu_depth_data_labeled.mat')

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
    np.save('/home/dataset2/nyu/nyu1/'+str(i)+'.npy',group)

