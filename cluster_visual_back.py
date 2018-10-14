# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-07-31 20:35:41
# @Last Modified by:   yulidong
# @Last Modified time: 2018-08-07 21:21:19
import os
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import time
import cv2
import torch
COLOR=[np.array([255,0,0]), 
       np.array([0,255,0]),
       np.array([0,0,255]),
       np.array([125,125,0]),
       np.array([0,125,125]),
       np.array([125,0,125]),
       np.array([50,100,50]),
       np.array([100,50,100])
       ]

def cluster(prediction, bandwidth):
    ms = MeanShift(bandwidth, bin_seeding=True)
    print ('Mean shift clustering, might take some time ...')
    tic = time.time()
    ms.fit(prediction)
    print ('time for clustering', time.time() - tic)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    
    num_clusters = cluster_centers.shape[0]

    return num_clusters, labels, cluster_centers

def get_instance_masks(prediction, bandwidth):
    batch_size, h, w, feature_dim = prediction.shape

    instance_masks = []
    for i in range(batch_size):
        num_clusters, labels, cluster_centers = cluster(prediction[i].reshape([h*w, feature_dim]), bandwidth)
        print ('Number of predicted clusters', num_clusters)
        labels = np.array(labels, dtype=np.uint8).reshape([h,w])
        mask = np.zeros([h,w], dtype=np.uint8)

        num_clusters = min([num_clusters,8])
        for mask_id in range(num_clusters):
            ind = np.where(labels==mask_id)
            mask[ind] = np.uint8(255/num_clusters*mask_id)


        instance_masks.append(mask)

    return np.array(instance_masks)


def save_instance_masks(prediction,output_dir, bandwidth, count):
    batch_size, h, w, feature_dim = prediction.shape

    instance_masks = []
    for i in range(batch_size):
        num_clusters, labels, cluster_centers = cluster(prediction[i].reshape([h*w, feature_dim]), bandwidth)
        print ('Number of predicted clusters', num_clusters)
        labels = np.array(labels, dtype=np.uint8).reshape([h,w])
        mask = np.zeros([h,w,3], dtype=np.uint8)

        num_clusters = min([num_clusters,8])
        for mask_id in range(num_clusters):
            mask = np.zeros([h,w,3], dtype=np.uint8)
            ind = np.where(labels==mask_id)
            mask[ind] = np.array([255,255,255])
            output_file_name = os.path.join(output_dir, 'cluster_{}_{}.png'.format(str(count).zfill(4), str(mask_id)))
            cv2.imwrite(output_file_name, mask)


        instance_masks.append(mask)

    return instance_masks

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
    #print(torch.sum(mask))
    if torch.sum(mask)<400:
        return mask.float(),torch.tensor(0).float().cuda()
    minx=torch.min(pixels[:,0])
    maxx=torch.max(pixels[:,0])
    miny=torch.min(pixels[:,1])
    maxy=torch.max(pixels[:,1])
    #areas=torch.ceil((maxx-minx).float()/60)*torch.ceil((maxy-miny).float()/60)
    for i in range(1,torch.ceil((maxx-minx).float()/60).int()+1):
        for j in range(1,torch.ceil((maxy-miny).float()/60).int()+1):
            if torch.sum(mask[minx+60*(i-1):minx+60*i,miny+60*(j-1):miny+60*j])>400:
                mask[minx+60*(i-1):minx+60*i,miny+60*(j-1):miny+60*j]*=i*j

    areas=torch.unique(mask).sort()[0]
    for i in range(1,len(areas)):
        mask=torch.where(mask==areas[i],-torch.tensor(i).cuda(),mask)
    mask=-mask
    areas=len(areas)-1
    #print(torch.sum(mask))
    return mask.float(),torch.tensor(areas).float().cuda()

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
def fast_cluster(feature,bandwidth=0.7):
    masks=[]
    areas=[]
    segments=[]
    for i in range(feature.shape[0]):
        n_feature=feature[i,...]
        label=torch.zeros(n_feature.shape[1],n_feature.shape[2]).cuda().float()
        n_masks=[]
        n_areas=[]
        while(torch.min(label)==0):
            candidate=torch.where(label==0,torch.tensor(1).float().cuda(),torch.tensor(0).float().cuda()).nonzero()
            #print(len(candidate))
            seed=torch.randint(len(candidate),(1,))[0].long()
            mean=n_feature[:,candidate[seed][0].long(),candidate[seed][1].long()].view(n_feature.shape[0],1,1)
            mean=mean_shift(n_feature, mean, bandwidth)
            t_masks,t_areas=get_mask(n_feature, mean, bandwidth)
            #print(torch.sum(t_masks))
            label=label+t_masks
            #print(torch.sum(label))
            if t_areas>0:
                n_masks.append(t_masks)
                n_areas.append(t_areas)
        mask,count,region=re_label(torch.stack(n_masks),torch.stack(n_areas),bandwidth=0.7)
        masks.append(mask)
        areas.append(count)
        segments.append(region)
    masks=torch.stack(masks)
    areas=torch.stack(areas)
    segments=torch.stack(segments)
    return masks,areas,segments
