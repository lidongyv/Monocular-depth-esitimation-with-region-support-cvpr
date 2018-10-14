# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-04-12 12:34:57
# @Last Modified by:   yulidong
# @Last Modified time: 2018-04-18 11:08:32
import scipy.io
import numpy as np
data=scipy.io.loadmat('/home/lidong/Documents/datasets/nyu/train_test.mat')
print('load success')
train_depth=data['train_depth']
print(train_depth.shape)
train_image=data['train_image']
print(train_image.shape)
test_depth=data['test_depth']
test_image=data['test_image']
num=train_depth.shape[-1]
train_depth=np.reshape(train_depth,[480,640,1,num])
num=test_depth.shape[-1]
test_depth=np.reshape(test_depth,[480,640,1,num])
train=np.concatenate((train_image,train_depth),2)
test=np.concatenate((test_image,test_depth),2)

test_depth=[]
test_image=[]
train_depth=[]
train_image=[]
train=np.concatenate((train,test),3)

print('free')
data=scipy.io.loadmat('/home/lidong/Documents/datasets/nyu/nyu2.mat')
print('load success')
depths=data['depths']
images=data['images']
num=depths.shape[-1]
depths=np.reshape(depths,[480,640,1,num])
data=np.concatenate((images,depths),2)
depths=[]
images=[]
testn=np.random.randint(0,num,549)
test=data[:,:,:,testn[0]]
test=np.reshape(test,[480,640,4,1])
train2=data[:,:,:, 0]
train2=np.reshape(train2,[480,640,4,1])
for i in range(num):
    print(i)
    temp=np.reshape(data[:,:,:,i],[480,640,4,1])
    if i in testn:
        test=np.concatenate((test,temp),3)
    else:
        train2=np.concatenate((train2,temp),3)

data=[]
print('free')
#train=np.concatenate((train,train2),3)
print(train.shape)
print(test.shape)
np.save('/home/lidong/Documents/datasets/nyu/nyu1_all.npy',train)
np.save('/home/lidong/Documents/datasets/nyu/nyu2_train.npy',train2)
np.save('/home/lidong/Documents/datasets/nyu/nyu2_test.npy',test)     
print('done')