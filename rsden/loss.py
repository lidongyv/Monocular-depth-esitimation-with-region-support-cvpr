# -*- coding: utf-8 -*-
# @Author: lidong
# @Date:   2018-03-18 16:31:14
# @Last Modified by:   yulidong
# @Last Modified time: 2018-08-07 22:36:47

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def l1_r(input, target, weight=None, size_average=True):
    relation=[]
    loss=nn.MSELoss()

    for i in range(3):
        target=torch.reshape(target,(input[i].shape))
        #print(target.shape)
        t=loss(input[i],target)
        #print(t.item())
        relation.append(t)

    return relation
def l1_a(input, target, weight=None, size_average=True):
    relation=[]
    loss=nn.MSELoss()

    for i in range(4):
        target=torch.reshape(target,(input[i].shape))
        #print(target.shape)
        t=torch.sqrt(loss(input[i],target))
        #print(t.item())
        relation.append(t)

    return relation
def log_r(input, target, weight=None, size_average=True):
    relation=[]
    d=[]
    out=[]
    target=torch.reshape(target,(input[0].shape))
    target=torch.log10(target+1e-6)
    loss=nn.MSELoss()
    for i in range(3):
        # pre=input[i]
        # num=torch.sum(torch.where(pre>0,torch.ones_like(pre),torch.zeros_like(pre)))/torch.sum(torch.ones_like(pre))
        # print(num)
        input[i]=torch.log10(input[i]+1e-6)  
        relation.append(loss(input[i],target))
        #d.append(0.5*torch.pow(torch.sum(input[i]-target),2)/torch.pow(torch.sum(torch.ones_like(input[i])),2))
        #out.append(relation[i]-d[i])
    return relation

def log_kitti(input, target, weight=None, size_average=True):
    zero=torch.zeros_like(input)
    target=torch.reshape(target,(input.shape))
    loss=nn.MSELoss(size_average=False) 
    input=torch.where(target>0,torch.log10(input),zero)
    target=torch.where(target>0,torch.log10(target),zero)

    #relation=torch.sqrt(loss(input,target)) 
    relation=loss(input,target)/torch.sum(torch.where(target>0,torch.ones_like(input),zero))
    d=0.5*torch.pow(torch.sum(input-target),2)/torch.pow(torch.sum(torch.where(target>0,torch.ones_like(input),zero)),2)
 
def log_r_kitti(input, target, weight=None, size_average=True):

    relation=[]
    d=[]
    out=[]
    target=torch.reshape(target,(input[0].shape))
    zero=torch.zeros_like(target)
    target=torch.where(target>0,torch.log10(target),zero)
    loss=nn.MSELoss(size_average=False)
    one=torch.ones_like(target)
    num=torch.sum(torch.where(target>0,one,zero))
    for i in range(3):
        pre=torch.where(target>0,torch.log10(input[i]),zero)
        relation.append(loss(pre,target)/num)
    return relation 
def mask_depth_loss(depth,mask,target,segment):
    mask=mask.float()
    target=target.float()
    depth=depth.float()
    segment=segment.float()
    #mse=nn.MSELoss(size_average=False)
    mask_map=torch.argmax(mask,dim=1)
    one=torch.ones_like(mask_map).float()
    zeros=torch.zeros_like(mask_map).float()
    mask_map_r=torch.where(mask_map==1,one,zeros)
    mask_ground=torch.where(segment==1,one,zeros)
    #print(mask_ground.device)
    #print(target.device)
    loss=torch.pow(torch.sum(mask_map_r*depth)/(torch.sum(mask_map_r)+1)-torch.sum(mask_ground*target)/(torch.sum(mask_ground)+1),2)
    region=torch.zeros_like(mask_map_r)     
    for i in range(2,torch.max(mask_map.int())+1):
        mask_map_r=torch.where(mask_map==i,one,zeros)
        mask_ground=torch.where(segment==1,one,zeros)
        loss+=torch.pow(torch.sum(mask_map_r*depth)/(torch.sum(mask_map_r)+1)-torch.sum(mask_ground*target)/(torch.sum(mask_ground)+1),2)
        region+=mask_map_r*torch.sum(mask_map_r*depth)/(torch.sum(mask_map_r)+1)
    #print(loss)       
    return loss/torch.max(mask_map).float(),region
def region_generation(depth,mask,target,segment):
    mask=mask.float()
    target=target.float()
    target=torch.log10(target+1e-12) 
    depth=depth.float()

    segment=segment.float()
    #mse=nn.MSELoss(size_average=False)
    mask_map=torch.argmax(mask,dim=1).float()
    ones=torch.ones_like(mask_map).float()
    zeros=torch.zeros_like(mask_map).float()
    region=torch.zeros_like(mask_map)     
    for i in range(0,torch.max(mask_map.int())+1):
        mask_map_r=torch.where(mask_map==i,ones,zeros)
        region+=mask_map_r*torch.sum(mask_map_r*depth)/torch.max(torch.sum(mask_map_r),torch.ones_like(torch.sum(mask_map_r)))
    return region
def region_loss(depth,mask,target,segment):
    mse=nn.MSELoss(reduction='none')
    
    mask=mask.float()
    target=target.float()
    target=torch.log10(target+1e-12) 
    depth=depth.float()
    segment=segment.float()
    #mse=nn.MSELoss(size_average=False)
    mask_map=torch.argmax(mask,dim=1).float()
    ones=torch.ones_like(mask_map).float()
    zeros=torch.zeros_like(mask_map).float()
    region=torch.zeros_like(mask_map).float()   
    for i in range(0,torch.max(mask_map.int())+1):
        mask_map_r=torch.where(mask_map==i,ones,zeros)
        region+=mask_map_r*torch.sum(mask_map_r*depth)/torch.max(torch.sum(mask_map_r),torch.ones_like(torch.sum(mask_map_r)))
    region_l1=region
    region=torch.log10(region+1e-12)
    target=torch.reshape(target,region.shape)
    dis=mse(region,target)
    for i in range(1,torch.max(segment.int())+1):
        mask_map_r=torch.where(segment==i,ones,zeros)
        if i==1:
            loss=torch.sum(mask_map_r*dis)/(torch.sum(mask_map_r)+1)
        else:
            loss+=torch.sum(mask_map_r*dis)/(torch.sum(mask_map_r)+1)
    return loss/torch.max(segment).float(),region_l1
def mask_loss(input,target):
    target=torch.reshape(target,[input.shape[0],input.shape[2],input.shape[3]]).long()
    #print(torch.max(target))
    #print(input.shape)
    nll=torch.nn.NLLLoss()
    loss=nll(input,target)
    #print(loss)
    return loss
def mask_loss_region(mask,segment):

    segment=torch.reshape(segment,[mask.shape[0],mask.shape[2],mask.shape[3]]).float()
    ones=torch.ones_like(segment).float()
    zeros=torch.zeros_like(segment).float()
    mask_map_v=torch.argmax(mask,dim=1).float()+ones
    #mask_map_p=torch.max(mask,dim=1).float()
    ground=torch.ones_like(segment)
    for i in range(1,torch.max(segment.int())+1):
        mask_r=torch.where(segment==i,mask_map_v,zeros)
        mask_r_o=torch.where(segment==i,ones,zeros)
        mask_o=mask_map_v-mask_r
        mask_o_o=ones-mask_r_o
        value=torch.bincount(torch.reshape(mask_r,(torch.sum(ones),)).int())[1:]
        mean=torch.argmax(value).float()+1
        mask_p=torch.exp(mask[:,mean.long(),:,:])
        #print(torch.max(mask_p))
        mask_p_r=torch.where(segment==i,mask_p,zeros)
        inner=-0.25*torch.log10(torch.sum(mask_p_r)/(torch.sum(mask_r_o)+1)+1e-12) \
              -0.25*torch.log10(torch.pow(torch.sum(mask_p_r)/torch.sum(mask_p),2)) 
              #-0.25*torch.log10(torch.sum(mask_p_r)/torch.sum(mask_p)) 
        #print(1-torch.sum(torch.where(mask_o==mean,mask_p,zeros))/(torch.sum(mask_o_o)+1))
        outer=-torch.log10(1-torch.sum(mask_p-mask_p_r)/(torch.sum(mask_o_o)+1)+1e-12) \
              -torch.log10(torch.pow(1-torch.sum(mask_p-mask_p_r)/torch.sum(mask_p),2)) 
              #-torch.log10(1-torch.sum(mask_p-mask_p_r)/torch.sum(mask_p)) 
        #print(outer)
        #exit()
        #print(outer)
        if i==1:
            loss=inner+outer
        else:
            loss+=inner+outer       
        ground=torch.where(segment==i,ground*mean,ground)
    loss=loss/torch.max(segment)
    #nll=torch.nn.NLLLoss()
    #loss_nll=nll(mask,ground.long())
    return loss
def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    #print(c,target.max().data.cpu().numpy())

    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n * h * w, 1).repeat(1, c) >= 0]
    log_p = log_p.view(-1, c)

    mask = target >= 0
    target = target[mask]
    #loss=log_p.sum()
    loss = F.nll_loss(log_p, target,
                      weight=weight, size_average=False)
    #print(loss, mask.data.sum())
    if size_average:
    #    print(mask.data.sum())
       loss /= mask.data.sum()
    #    loss=loss/(950*540)
    return loss
def l1(input, target, weight=None, size_average=True):
    target=torch.reshape(target,(input.shape))
    loss=nn.MSELoss()
    relation=torch.sqrt(loss(input,target))
    #mean=torch.abs(torch.mean(input)-torch.mean(target))
    #output=relation+0.2*mean
    return relation
def l2(input, target, weight=None, size_average=True):
    target=torch.reshape(target,(input.shape)).float()
    loss=nn.MSELoss() 
    #relation=torch.sqrt(loss(input,target)) 
    relation=loss(input,target) 

    return relation
def berhu(input,target):
    target=torch.reshape(target,(input.shape)).float()
    l1=torch.nn.L1Loss(reduction='none')
    l2=torch.nn.MSELoss(reduction='none')
    l1_loss=l1(input,target)
    l2_loss=l2(input, target)
    c=torch.max(l1_loss)/5
    l2=(l2_loss+torch.pow(c,2))/(2*c)
    loss=torch.mean(torch.where(torch.le(l1_loss,c),l1_loss,l2_loss))
    return loss
def log_loss(input, target, weight=None, size_average=True):
    # num=torch.sum(torch.where(input==0,torch.ones_like(input),torch.zeros_like(input)))
    # positive=num/torch.sum(toerch.ones_like(input))e
    # print(positive.item())
    target=torch.reshape(target,(input.shape)).float()
    loss=nn.MSELoss() 
    input=torch.log10(input+1e-12) 
    target=torch.log10(target+1e-12) 
    #relation=torch.sqrt(loss(input,target)) 
    relation=loss(input,target) 
    #d=0.5*torch.pow(torch.sum(input-target),2)/torch.pow(torch.sum(torch.ones_like(input)),2)
    #relation=relation-d 
    return relation

    # target=torch.reshape(target,(input.shape))
    # #loss=nn.MSELoss()
    # num=torch.sum(torch.where(input>0,torch.ones_like(input),torch.zeros_like(input)))
    # input=torch.log10(torch.where(input>0,input,torch.ones_like(input)))
    # target=torch.log10(torch.where(target>0,target,torch.ones_like(target)))
    # # #relation=torch.sqrt(loss(input,target))
    # relation=torch.sum(torch.pow(torch.where(input==0,input,input-target),2))/num
    # d=torch.pow(torch.sum(torch.where(input==0,input,input-target)),2)/torch.pow(num,2)*0.5
    # #positive=num/torch.sum(torch.ones_like(input))
    # #print(positive.item())
    # #-torch.sum(torch.where(input<0,input,torch.zeros_like(input)))/num
    # losses=relation+d
    # return losses

def log_l1(input, target, weight=None, size_average=True):
    l1loss=l1(input,target)
    logloss=log_loss(input,target)
    num=torch.sum(torch.where(input==0,torch.ones_like(input),torch.zeros_like(input)))
    positive=num/torch.sum(torch.ones_like(input))
    print(positive.item())
    loss=(1-positive)*logloss+positive*l1loss
    return loss
def l1_kitti(input, target, weight=None, size_average=True):
    zero=torch.zeros_like(input)
    target=torch.reshape(target,(input.shape))
    input=torch.where(target>0,input,zero)
    target=torch.where(target>0,target,zero)
    loss=nn.MSELoss(size_average=False) 
    relation=loss(input,target)/torch.sum(torch.where(target>0,torch.ones_like(input),zero))
    return relation
def log_kitti(input, target, weight=None, size_average=True):
    zero=torch.zeros_like(input)
    target=torch.reshape(target,(input.shape))
    loss=nn.MSELoss(size_average=False) 
    input=torch.where(target>0,torch.log10(input),zero)
    target=torch.where(target>0,torch.log10(target),zero)

    #relation=torch.sqrt(loss(input,target)) 
    relation=loss(input,target)/torch.sum(torch.where(target>0,torch.ones_like(input),zero))
    d=0.5*torch.pow(torch.sum(input-target),2)/torch.pow(torch.sum(torch.where(target>0,torch.ones_like(input),zero)),2)
 
    return relation-d 
# def region(input,target,instance):
#     loss=0
#     lf=nn.MSELoss(size_average=False,reduce=False)
#     target=torch.reshape(target,(input.shape))
#     instance=torch.reshape(instance,(input.shape))
#     zero=torch.zeros_like(input)
#     one=torch.ones_like(input)
#     dis=lf(input,target)
#     for i in range(0,int(torch.max(instance).item()+1)):
#         input_region=torch.where(instance==i,input,zero)
#         ground_region=torch.where(instance==i,target,zero)
#         m=torch.max(ground_region)
#         if m==0:
#             continue
#         num=torch.sum(torch.where(instance==i,one,zero))
#         loss+=lf(input_region,ground_region)/num
#         # average=torch.sum(input_region)/num
#         # input_region=input_region-average
#         # input_region=torch.pow(input_region,2)
#         # var=torch.sum(input_region)/num
#         # loss+=0.5*var
#     loss=loss/torch.max(instance)
#     return loss


def region(input,target,instance):
    loss=0
    lf=nn.MSELoss(size_average=False,reduce=False)
    target=torch.reshape(target,(input.shape))
    # input=torch.log10(input+1e-12) 
    # target=torch.log10(target+1e-12) 
    instance=torch.reshape(instance,(input.shape))
    zero=torch.zeros_like(input)
    one=torch.ones_like(input)
    dis=lf(input,target)
    for i in range(1,int(torch.max(instance).item()+1)):
        dis_region=torch.where(instance==i,dis,zero)
        num=torch.sum(torch.where(instance==i,one,zero))
        average=torch.sum(dis_region)/num
        loss=loss+average
        #dis_region=torch.where(instance==i,dis_region-average,zero)
        # var=0.1*torch.sqrt(torch.sum(torch.pow(dis_region,2))/num)/average
        # loss=loss+var
    loss=loss/(torch.max(instance))
    return loss

def region_log(input,target,instance):
    loss=0
    lf=nn.MSELoss(reduction='none')
    target=torch.reshape(target,(input.shape))
    input=torch.log10(input+1e-6) 
    target=torch.log10(target+1e-6) 
    instance=torch.reshape(instance,(input.shape)).float()
    zero=torch.zeros_like(input)
    one=torch.ones_like(input)
    dis=lf(input,target)
    for i in range(1,int(torch.max(instance).item()+1)):
        dis_region=torch.where(instance==i,dis,zero)
        num=torch.sum(torch.where(instance==i,one,zero))
        average=torch.sum(dis_region)/num
        loss=loss+average
        # dis_region=torch.where(instance==i,dis_region-average,zero)
        # var=(torch.sum(torch.pow(dis_region,2))/num)/average
        # loss=loss+var
    loss=loss/(torch.max(instance))
    #print(torch.max(instance).item())
    return loss


def region_r(input,target,instance):
    loss=0
    relation=[]
    lf=nn.MSELoss(size_average=False,reduce=False)
    target=torch.reshape(target,(input[0].shape))
 
    target=torch.log10(target+1e-6) 
    instance=torch.reshape(instance,(input[0].shape))
    zero=torch.zeros_like(input[0])
    one=torch.ones_like(input[0])
    for i in range(3):
        input[i]=torch.log10(input[i]+1e-6)
        dis=lf(input[i],target)
        for i in range(1,int(torch.max(instance).item()+1)):
            dis_region=torch.where(instance==i,dis,zero)
            num=torch.sum(torch.where(instance==i,one,zero))
            average=torch.sum(dis_region)/num
            loss=loss+average
        #print(torch.max(instance).item())
        relation.append(loss/(torch.max(instance)))
        loss=0
    return relation