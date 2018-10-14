# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-04-26 10:28:48
# @Last Modified by:   yulidong
# @Last Modified time: 2018-10-10 23:22:28
import os
import numpy as np
import cv2
import copy
def seg2instance(seg):
    region=seg
    label=-1
    while(np.max(seg)>0):
        category=np.max(seg)
        while(np.max(seg)==category):
            positions=np.array(np.where(seg==category))

            region[positions[0,0],positions[1,0]]=label
            for i in range(1,positions.shape[1]):
                if np.abs(positions[0,i]-positions[0,i-1])+np.abs(positions[1,i]-positions[1,i-1])<=2:
                     region[positions[0,i],positions[1,i]]=label
                elif region[positions[0,i],max(0,positions[1,i]-1)]==label or\
                        region[max(0,positions[0,i]-1),min(positions[1,i]+1,639)]==label or\
                        region[max(0,positions[0,i]-1),positions[1,i]]==label or\
                        region[max(0,positions[0,i]-1),max(0,positions[1,i]-1)]==label:
                        region[positions[0,i],positions[1,i]]=label
                else:
                    break;
        label=label-1
    return np.abs(region)

def reg2supportd(reg,dep,variance):
    supportd=np.zeros([reg.shape[0],reg.shape[1]])
    for i in range(1,int(np.max(reg)+1)):
        region=np.where(reg==i)
        depth=dep[region]
        if(np.max(depth)-np.min(depth)<2*variance):
            supportd[region]=np.mean(depth)

        else:
            start=np.min(depth)
            end=np.min(depth)+variance
            while(end<np.max(depth)):
                region1=np.where(np.logical_and(reg==i,np.logical_and(dep>start,dep<end)))
                if len(dep[region1])>len(depth)/((np.max(depth)-np.min(depth))/variance):
                    supportd[region1]=np.mean(dep[region1])

                    if end+variance<np.max(depth):
                        start=end
                        end=end+variance
                    else:
                        region1=np.where(np.logical_and(reg==i,np.logical_and(dep>start,dep<np.max(depth))))
                        supportd[region1]=np.mean(dep[region1])
                        break;
                else:

                    if end+variance<np.max(depth):
                        start=start
                        end=end+variance
                    else:
                        region1=np.where(np.logical_and(reg==i,np.logical_and(dep>start,dep<np.max(depth))))
                        supportd[region1]=np.mean(dep[region1])
                        break;                    
    return supportd                    
load_path='/home/dataset2/nyu/nyu1/all'
save_path='/home/dataset2/nyu/nyu1/train'
filenames=os.listdir(load_path)
filenames.sort(key=lambda x:int(x[:-4]))
for i in range(len(filenames)):
    data=np.load(os.path.join(load_path,filenames[i]))
    # if data.shape[2]>5:
    #     np.save(os.path.join(load_path,filenames[i]),data[:,:,0:5])    
    data=data[:,:,0:5]
    img = data[:,:,0:3]
    depth = data[:,:,3]
    segments = data[:,:,4]    
    instance=seg2instance(copy.deepcopy(segments))
    variance=(np.max(depth)-np.min(depth))/10
    # print(variance)
    # exit()
    supportd=reg2supportd(copy.deepcopy(instance),copy.deepcopy(depth),variance)
    supporti=seg2instance(copy.deepcopy(supportd))
    # data=np.concatenate((data,np.reshape(instance,[instance.shape[0],instance.shape[1],1])),2)
    # data=np.concatenate((data,np.reshape(supportd,[supportd.shape[0],supportd.shape[1],1])),2)
    # print(np.max(instance),np.max(instance),np.max(instance))
    # a=np.reshape(supportd/np.max(supportd),[supportd.shape[0],supportd.shape[1],1])
    # b=np.reshape(instance/np.max(instance),[supportd.shape[0],supportd.shape[1],1])
    # c=np.reshape(supporti/np.max(supporti),[supporti.shape[0],supporti.shape[1],1])
    # regionsupport_vision=np.concatenate((a,b),2)
    # regionsupport_vision=np.concatenate((regionsupport_vision,c),2)
    supportd=np.reshape(supportd,[supportd.shape[0],supportd.shape[1],1])
    instance=np.reshape(instance,[instance.shape[0],instance.shape[1],1])
    supporti=np.reshape(supporti,[supporti.shape[0],supporti.shape[1],1])
    region_support=np.concatenate((supportd,instance),2)
    region_support=np.concatenate((region_support,supporti),2)
    data=np.concatenate((data,region_support),2)
    print(os.path.join(save_path,filenames[i]))
    np.save(os.path.join(save_path,filenames[i]),data)

    # cv2.namedWindow('show')
    # cv2.imshow('show',regionsupport_vision)
    # # cv2.namedWindow('show2')
    # # cv2.imshow('show2',b)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
