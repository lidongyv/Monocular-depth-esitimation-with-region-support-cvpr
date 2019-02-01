# -*- coding: utf-8 -*-
# @Author: lidong
# @Date:   2018-03-18 16:31:14
# @Last Modified by:   yulidong
# @Last Modified time: 2019-01-29 15:02:01

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
alpha=0.7132995128631592
beta=9.99547004699707
def cluster_loss(feature,segment,lvar=0.16,dis=0.34,device_id=0):
    lvar=torch.tensor(lvar).float().cuda(device_id)
    dis=torch.tensor(dis).float().cuda(device_id)
    #segment=torch.squeeze(segment)
    instance_num=torch.max(segment)
    #print(instance_num)
    ones=torch.ones_like(segment).float()
    zeros=torch.zeros_like(segment).float()
    mean=[]
    #var=[]
    #print(feature.shape)
    #print(segment.shape)
    for i in range(1,torch.max(segment)+1):
        mask_r=torch.where(segment==i,ones,zeros)
        feature_r=feature*mask_r
        count=torch.sum(mask_r)
        if count==0:
            count=1
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
    loss_var=loss_var/instance_num.float().cuda(device_id)
    mean=torch.stack(mean)
    #mean shape instance_num*N*C
    #var=torch.mean(torch.stack(var))
    #mean1-mean1
    #mean2-mean1
    #mean3-mean1
    #mean1-mean2e
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

    if torch.isnan(loss_var):
        print(loss_var,loss_dis,loss_reg)
        print(instance_num,count)
        exit()
    #exit()
    return loss_var,loss_dis,loss_reg


def cluster_loss_depth(feature,segment,depth,lvar=0.16,dis=0.34,device_id=2):
    segment=segment.view(1,1,segment.shape[-2],segment.shape[-1])
    lvar=torch.tensor(lvar).float().cuda(device_id)
    depth=depth.view_as(segment)
    d_var=((alpha-beta)/200*torch.ones(1)).cuda(device_id)
    d_var=torch.pow(d_var/2,2)
    dis=torch.tensor(dis).float().cuda(device_id)
    instance_num=torch.max(segment)
    ones=torch.ones_like(segment).float()
    zeros=torch.zeros_like(segment).float()
    mean=[]
    depth_mask=(depth>alpha)&(depth<beta)
    depth_mask=depth_mask.float()
    depth=depth*depth_mask
    for i in range(1,torch.max(segment)+1):
        mask_r=(segment==i).float()
        mask_r_d=mask_r*depth_mask
        depth_r=depth*mask_r_d
        count=torch.sum(mask_r_d)
        if count==0:
            count=1
        #print(count)
        mean_r_d=torch.sum(depth_r)/count
        var_r_d=torch.where(mask_r_d==ones,torch.pow(depth_r-mean_r_d,2),zeros)
        #check the real mask
        mask_i=(var_r_d<=d_var).float()*mask_r_d
        depth_r2=depth*mask_i
        count=torch.sum(mask_i)
        if count==0:
            count=1
        #print(count)
        mean_r_d=torch.sum(depth_r2)/count
        var_r_d=torch.where(mask_r_d==ones,torch.pow(depth_r-mean_r_d,2),zeros)
        mask_i=(var_r_d<=d_var).float()*mask_r_d
        count=torch.sum(mask_i)
        #print(count)
        if count==0:
            count=1
        feature_i=feature*mask_i
        mean_r=torch.sum(torch.sum(feature_i,dim=-1),dim=-1)/count

        mean_r_volume=mean_r.view(mean_r.shape[0],mean_r.shape[1],1,1).expand(1,-1,feature.shape[-2],feature.shape[-1])
        var_map=torch.where(mask_i==ones,torch.norm(feature_i-mean_r_volume,dim=1),zeros)
        loss_var_r=torch.where(mask_i==ones,var_map-lvar,zeros)
        if i==1:
            loss_var=torch.sum(torch.pow(torch.clamp(loss_var_r,min=0),2))/count
        else:
            loss_var+=torch.sum(torch.pow(torch.clamp(loss_var_r,min=0),2))/count
        mask_o=torch.where(var_r_d>d_var,ones,zeros)*mask_r_d
        count=torch.sum(mask_o)
        #print(count)
        if count==0:
            count=1
        feature_o=feature*mask_o
        # mean_r=torch.sum(torch.sum(feature_r,dim=-1),dim=-1)/count
        # mean_r_volume=mean_r.view(mean_r.shape[0],mean_r.shape[1],1,1).expand(-1,-1,feature.shape[-2],feature.shape[-1])
        var_map=torch.where(mask_o==ones,torch.norm(feature_o-mean_r_volume,dim=1),zeros)
        loss_var_r=torch.where(mask_o==ones,2*lvar-var_map,zeros)
        if i==1:
            loss_dis=torch.sum(torch.pow(torch.clamp(loss_var_r,min=0),2))/count
            #print(loss_dis,count)
        else:
            #print(torch.sum(torch.pow(torch.clamp(loss_var_r,min=0),2))/count,count)
            loss_dis+=torch.sum(torch.pow(torch.clamp(loss_var_r,min=0),2))/count
        mask_r_d=torch.where(torch.pow(depth-mean_r_d,2)<=d_var,ones,zeros)*depth_mask
        count=torch.sum(mask_r_d)
        if count==0:
            count=1
        feature_r_d=feature*mask_r_d
        var_map=torch.where(mask_r_d==ones,torch.norm(feature_r_d-mean_r_volume,dim=1),zeros)
        loss_var_r=torch.where(mask_r_d==ones,var_map-lvar,zeros)
        if i==1:
            loss_re=torch.sum(torch.pow(torch.clamp(loss_var_r,min=0),2))/count
        else:
            loss_re+=torch.sum(torch.pow(torch.clamp(loss_var_r,min=0),2))/count

    count=torch.max(segment)
    if count==0:
        count=1
    loss_var=loss_var/count
    loss_dis=loss_dis/count
    loss_re=loss_re/count
    loss_nan=loss_var+loss_dis+loss_re
    # print(loss_var,loss_dis,loss_re)
    # exit()
    if torch.isnan(loss_nan) or torch.isinf(loss_nan):
        print(loss_var,loss_dis,loss_re)
        print(instance_num,count)
        exit()
    #exit()
    return loss_var,loss_dis,loss_re
def semi_loss(feature,segment,lvar=0.16,dis=0.32,device_id=2):
    lvar=torch.tensor(lvar).float().cuda(device_id)
    dis=torch.tensor(dis).float().cuda(device_id)
    #segment=torch.squeeze(segment)
    instance_num=torch.max(segment)
    #print(instance_num)
    ones=torch.ones_like(segment).float()
    zeros=torch.zeros_like(segment).float()
    mean=[]
    #var=[]
    #print(feature.shape)
    #print(segment.shape)
    for i in range(1,torch.max(segment)+1):
        mask_r=torch.where(segment==i,ones,zeros)
        feature_r=feature*mask_r
        count=torch.sum(mask_r)
        if count==0:
            count=1
        mean_r=torch.sum(torch.sum(feature_r,dim=-1),dim=-1)/count
        mean.append(mean_r)
        #mean_r shape N*C
        mean_r_volume=mean_r.view(mean_r.shape[0],mean_r.shape[1],1,1).expand(-1,-1,feature.shape[-2],feature.shape[-1])
        var_map=torch.where(mask_r==ones,torch.norm(feature_r-mean_r_volume,dim=1),zeros)
        #print(var_map.shape)
        loss_var_r=var_map
        if i==1:
            loss_var=torch.sum(torch.pow(loss_var_r,2))/count
        else:
            loss_var+=torch.sum(torch.pow(loss_var_r,2))/count
        #var_r=torch.sum(var_map)/count
        #var.append(var_r)
    mean=torch.stack(mean)
    #instance_num=instance_num.float()
    loss_var=loss_var/instance_num.float().cuda(device_id)
    #print(mean.shape)
    # left=mean.view(1,instance_num,mean.shape[1],mean.shape[2]).expand(instance_num,instance_num,mean.shape[1],mean.shape[2])
    # right=mean.view(instance_num,1,mean.shape[1],mean.shape[2]).expand(instance_num,instance_num,mean.shape[1],mean.shape[2])
    # dis_map=torch.norm(left-right,dim=-1)
    # dis_map=torch.where(dis_map==0,torch.min(dis_map),dis_map)
    # dis=torch.min(dis_map)
    if torch.isnan(loss_var) or torch.isinf(loss_var):
        print(loss_var,loss_dis)
        print(instance_num,count)
        exit()
    #exit()
    return loss_var