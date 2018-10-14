# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-07-31 20:35:41
# @Last Modified by:   yulidong
# @Last Modified time: 2018-08-20 22:26:38
import os
import numpy as np
import time
import torch


def mean_shift(feature,mean,bandwidth):
    #feature shape c h w
    dis=feature-mean
    dis=torch.norm(dis,dim=0)
    mask=torch.where(dis<bandwidth,torch.tensor(1).cuda(),torch.tensor(0).cuda()).float()
    mean=torch.sum((feature*mask).view(feature.shape[0],feature.shape[1]*feature.shape[2]),dim=1)/torch.sum(mask)
    return mean
def get_mask(feature,mean,bandwidth):
    mean=mean.view([mean.shape[0],1,1])
    dis=feature-mean
    dis=torch.norm(dis,dim=0)
    mask=torch.where(dis<bandwidth,torch.tensor(1).cuda(),torch.tensor(0).cuda())
    pixels=mask.nonzero()
    return mask.float()


def re_label(mask,area,bandwidth):
    index=torch.sum(area)
    print(index)
    count=torch.tensor(0).float().cuda()
    for i in range(area.shape[0]):
        mask[i,:,:]=torch.where(mask[i,:,:]>0,mask[i,:,:]+count,mask[i,:,:])
        count+=area[i]
    segment=torch.where(mask>0,torch.tensor(1).cuda(),torch.tensor(0).cuda()).float()
    final=torch.sum(mask,dim=0)/torch.sum(segment,dim=0)
    final=torch.squeeze(final)
    final=final/255
    return mask,area,final
def refine_mask(t_mask):
    pixels=mask.nonzero()
    if torch.sum(mask)<400:
        return mask
    minx=torch.min(pixels[:,0])
    maxx=torch.max(pixels[:,0])
    miny=torch.min(pixels[:,1])
    maxy=torch.max(pixels[:,1])
    for i in range(1,torch.ceil((maxx-minx).float()/80).int()+1):
        for j in range(1,torch.ceil((maxy-miny).float()/80).int()+1):
            if torch.sum(mask[minx+80*(i-1):minx+80*i,miny+80*(j-1):miny+80*j])>400:
                mask[minx+80*(i-1):minx+80*i,miny+80*(j-1):miny+80*j]*=i*j
    areas=torch.unique(mask).sort()[0]
    for i in range(1,len(areas)):
        mask=torch.where(mask==areas[i],-torch.tensor(i).cuda(),mask)
    mask=-mask
    return mask.float()
def fuse_mask(n_mask,r_mask):
    base=torch.where(n_mask>0,torch.tensor(1).cuda(),torch.tensor(0).cuda()).float()
    areas=torch.max(n_mask)
    for i in range(1,torch.max(r_mask).long()+1):
        shift=torch.where(r_mask==i,torch.tensor(1).cuda(),torch.tensor(0).cuda()).float()
        overlap=torch.where(r_mask==i,shift-base,torch.tensor(0).cuda()).float()
        if torch.sum(overlap)/torch.sum(shift)>0.6:
            areas+=1
            n_mask=torch.where(overlap==1,areas,n_mask)
            base=torch.where(n_mask>0,torch.tensor(1).cuda(),torch.tensor(0).cuda()).float()
        else:
            area_num=torch.argmax(torch.bincount(torch.where(overlap==1,n_mask,torch.tensor(0).cuda()).long().view(-1))[1:])+1
            n_mask=torch.where(overlap==1,area_num,n_mask)
            base=torch.where(n_mask>0,torch.tensor(1).cuda(),torch.tensor(0).cuda()).float()
    areas_nums=torch.tensor(1).float().cuda()
    for i in range(1,torch.max(n_mask).long()+1):
        region=torch.where(n_mask==i,torch.tensor(1).cuda(),torch.tensor(0).cuda()).float()
        pixels=region.nonzero()
        minx=torch.min(pixels[:,0])
        maxx=torch.max(pixels[:,0])
        miny=torch.min(pixels[:,1])
        maxy=torch.max(pixels[:,1])
        for i in range(1,torch.ceil((maxx-minx).float()/80).int()+1):
            for j in range(1,torch.ceil((maxy-miny).float()/80).int()+1):
                if torch.sum(region[minx+80*(i-1):minx+80*i,miny+80*(j-1):miny+80*j])>400:
                    region[minx+80*(i-1):minx+80*i,miny+80*(j-1):miny+80*j]*=i*j
        areas=torch.unique(region).sort()[0]
        for i in range(1,len(areas)):
            region=torch.where(region==areas[i],-areas_nums,region)
            areas_nums+=1
        n_mask=torch.where(n_mask==i,region,n_mask)
    n_mask=-n_mask
    return n_mask
def fast_cluster(feature,bandwidth=0.16):
    masks=[]
    areas=[]
    segments=[]

    for i in range(feature.shape[0]):
        n_mask=0
        n_feature=feature[i,...]
        label=torch.zeros(n_feature.shape[1],n_feature.shape[2]).cuda().float()
        while(torch.min(label)==0):
            candidate=torch.where(label==0,torch.tensor(1).float().cuda(),torch.tensor(0).float().cuda()).nonzero()
            #print(len(candidate))
            seed=torch.randint(len(candidate),(1,))[0].long()
            mean=n_feature[:,candidate[seed][0].long(),candidate[seed][1].long()].view(n_feature.shape[0],1,1)
            mean=mean_shift(n_feature, mean, bandwidth)
            t_masks=get_mask(n_feature, mean, bandwidth)
            #print(torch.sum(t_masks))
            label=label+t_masks
            if n_mask==0:
                r_masks=refine_mask(t_mask)
                n_masks=t_masks
            else:
                r_masks=refine_mask(t_mask)
                n_masks=fuse_mask(n_mask,r_masks)

    return n_masks
