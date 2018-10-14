# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-04-16 20:31:45
# @Last Modified by:   yulidong
# @Last Modified time: 2018-04-16 21:51:27
import numpy as np
loss_rec=[]
for j in range(500):
    for i in range(500):
        loss_rec.append([j*500+i,i])
    np.save('/home/lidong/Documents/datasets/nyu/loss.npy',loss_rec) 