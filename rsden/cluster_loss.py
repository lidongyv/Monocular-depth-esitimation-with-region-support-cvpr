# -*- coding: utf-8 -*-
# @Author: lidong
# @Date:   2018-03-18 16:31:14
# @Last Modified by:   yulidong
# @Last Modified time: 2018-08-21 16:06:45

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def cluster_loss(feature,segment,lvar=0.16,dis=0.7):
    lvar=torch.tensor(lvar).float().cuda()
    dis=torch.tensor(dis).float().cuda()
    #segment=torch.squeeze(segment)
    instance_num=torch.max(segment)
    ones=torch.ones_like(segment).float()
    zeros=torch.zeros_like(segment).float()
    mean=[]
    #var=[]
    #print(feature.shape)
    #print(segment.shape)
    for i in range(1,instance_num+1):
        mask_r=torch.where(segment==i,ones,zeros)
        feature_r=feature*mask_r
        count=torch.sum(mask_r)
        mean_r=torch.sum(torch.sum(feature_r,dim=-1),dim=-1)/count
        #mean_r shape N*C
        mean_r_volume=mean_r.view(mean_r.shape[0],mean_r.shape[1],1,1).expand(-1,-1,feature.shape[-2],feature.shape[-1])
        mean.append(mean_r)
        var_map=torch.where(mask_r==ones,torch.norm(feature_r-mean_r_volume,dim=1),zeros)
        #print(var_map.shape)
        loss_var_r=var_map-lvar
        if i==1:
            loss_var=torch.sum(torch.pow(torch.clamp(loss_var_r,min=0),2))/count
        else:
            loss_var+=torch.sum(torch.pow(torch.clamp(loss_var_r,min=0),2))/count
        #var_r=torch.sum(var_map)/count
        #var.append(var_r)
    loss_var/=instance_num.float().cuda()
    mean=torch.stack(mean)
    #mean shape instance_num*N*C
    #var=torch.mean(torch.stack(var))
    #mean1-mean1
    #mean2-mean1
    #mean3-mean1
    #mean1-mean2
    #mean2-mean2
    #mean3-mean2
    #and mean123-mean3
    left=mean.view(1,instance_num,mean.shape[1],mean.shape[2]).expand(instance_num,instance_num,mean.shape[1],mean.shape[2])
    right=mean.view(instance_num,1,mean.shape[1],mean.shape[2]).expand(instance_num,instance_num,mean.shape[1],mean.shape[2])
    dis_map=torch.norm(left-right,dim=-1)
    #dis_map=torch.sum(dis_map,dim=-1)
    zeros=torch.zeros_like(dis_map)
    #print(dis_map.shape)
    instance_num=instance_num.float()
    loss_dis=torch.sum(torch.where(dis_map==zeros,dis_map,torch.pow(torch.clamp(2*dis-dis_map,min=0),2)))/(instance_num*instance_num-instance_num)
    #with mean shape instance_num*n*c
    loss_reg=torch.mean(torch.norm(mean,dim=-1))
    #exit()
    return loss_var,loss_dis,loss_reg
    #add the geometrical distance loss
    #two channel of the postion of pixels
    #x=torch.arange(0,feature.shape[2])
    #y=torch.arange(0,feature.shape[3])