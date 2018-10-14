# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-04-25 18:05:58
# @Last Modified by:   yulidong
# @Last Modified time: 2018-10-10 20:07:46
import matplotlib.pyplot as plt    
import numpy as np
import h5py
import os
mat_dir=r'/home/dataset/datasets/nyu2_depth/mat_data/'
mat_files=os.listdir(mat_dir)
mat_files.sort()
count=0
for i in range(len(mat_files)):
    mat_file=os.path.join(mat_dir,mat_files[i])
    print(mat_file)
    data =  h5py.File(mat_file)
    keys=[]
    values=[]
    #shapes=[]
    for k, v in data.items():
        keys.append(k)
        values.append(v)
        print(v)
    depths=data['depth']
    images=data['rgb']
    for j in range(depths.shape[-1]):
        image=images[...,j].astype('float')
        image=np.transpose(image,[2,1,0])
        depth=depths[...,j].astype('float')
        depth=np.transpose(depth,[1,0])
        depth=np.reshape(depth,[depth.shape[0],depth.shape[1],1])
        group=np.concatenate((image,depth),2)
        np.save('/home/dataset/datasets/nyu2_depth/npy_data/'+str(count)+'.npy',group)
        count+=1
        print(i,j,count)
    #exit()
# files=os.listdir('/home/lidong/Documents/datasets/nyu/train')
# start=len(files)
# files.sort(key=lambda x:int(x[:-4]))
#start=start-depths.shape[0]

