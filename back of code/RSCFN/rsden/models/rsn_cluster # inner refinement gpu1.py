# -*- coding: utf-8 -*-
# @Author: lidong
# @Date:   2018-03-20 18:01:52
# @Last Modified by:   yulidong
# @Last Modified time: 2018-10-31 21:48:19

import torch
import numpy as np
import torch.nn as nn
import math
from math import ceil
from torch.autograd import Variable
from rsden.cluster_loss import *
from rsden import caffe_pb2
from rsden.models.utils import *
cuda_id=2
group_dim=1
def mean_shift(feature,mean,bandwidth):
    #feature shape c h w
    for t in range(10):
        #print(t)
        dis=feature-mean
        dis=torch.norm(dis,dim=0)
        mask=torch.where(dis<bandwidth,torch.tensor(1).cuda(cuda_id),torch.tensor(0).cuda(cuda_id)).float()
        mean=torch.sum((feature*mask).view(feature.shape[0],feature.shape[1]*feature.shape[2]),dim=1)/torch.sum(mask)
        mean=mean.view([feature.shape[0],1,1])
    return mean
def get_mask(feature,mean,bandwidth):
    mean=mean.view([mean.shape[0],1,1])
    dis=feature-mean
    dis=torch.norm(dis,dim=0)
    mask=torch.where(dis<bandwidth,torch.tensor(1).cuda(cuda_id),torch.tensor(0).cuda(cuda_id))
    pixels=mask.nonzero()
    return mask.float()


def re_label(mask,area,bandwidth):
    index=torch.sum(area)
    print(index)
    count=torch.tensor(0).float().cuda(cuda_id)
    for i in range(area.shape[0]):
        mask[i,:,:]=torch.where(mask[i,:,:]>0,mask[i,:,:]+count,mask[i,:,:])
        count+=area[i]
    segment=torch.where(mask>0,torch.tensor(1).cuda(cuda_id),torch.tensor(0).cuda(cuda_id)).float()
    final=torch.sum(mask,dim=0)/torch.sum(segment,dim=0)
    final=torch.squeeze(final)
    final=final/255
    return mask,area,final
def refine_mask(mask):
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
        mask=torch.where(mask==areas[i],-torch.ones(1).float().cuda(cuda_id)*i,mask)
    mask=-mask
    return mask.float()
def fuse_mask(n_mask,r_mask):
    base=torch.where(n_mask>0,torch.tensor(1).cuda(cuda_id),torch.tensor(0).cuda(cuda_id)).float()
    areas=torch.max(n_mask)
    for i in range(1,torch.max(r_mask).long()+1):
        shift=torch.where(r_mask==i,torch.tensor(1).cuda(cuda_id),torch.tensor(0).cuda(cuda_id)).float()
        non_overlap=torch.where(base-shift==-1,torch.tensor(1).cuda(cuda_id),torch.tensor(0).cuda(cuda_id)).float()
        overlap=shift-non_overlap
        if torch.sum(non_overlap)/torch.sum(shift)>0.4:
            areas+=1
            n_mask=torch.where(non_overlap==1,areas,n_mask)
            base=torch.where(n_mask>0,torch.tensor(1).cuda(cuda_id),torch.tensor(0).cuda(cuda_id)).float()
            #print(areas)
        else:
            area_num=torch.argmax(torch.bincount(torch.where(overlap.long()==1,n_mask.long(),torch.tensor(0).cuda(cuda_id)).view(-1))[1:]).float()+1
            n_mask=torch.where(non_overlap==1,area_num,n_mask)
            base=torch.where(n_mask>0,torch.tensor(1).cuda(cuda_id),torch.tensor(0).cuda(cuda_id)).float()
            #print(areas)
#     areas_nums=torch.tensor(1).float().cuda(cuda_id)
#     for i in range(1,torch.max(n_mask).long()+1):
#         region=torch.where(n_mask==i,torch.tensor(1).cuda(cuda_id),torch.tensor(0).cuda(cuda_id)).float()
#         pixels=region.nonzero()
#         if pixels.shape[0]>0:
#             minx=torch.min(pixels[:,0])
#             maxx=torch.max(pixels[:,0])
#             miny=torch.min(pixels[:,1])
#             maxy=torch.max(pixels[:,1])
#             for i in range(1,torch.ceil((maxx-minx).float()/80).int()+1):
#                 for j in range(1,torch.ceil((maxy-miny).float()/80).int()+1):
#                     if torch.sum(region[minx+80*(i-1):minx+80*i,miny+80*(j-1):miny+80*j])>400:
#                         region[minx+80*(i-1):minx+80*i,miny+80*(j-1):miny+80*j]*=i*j
#             areas=torch.unique(region).sort()[0]
#             for i in range(1,len(areas)):
#                 region=torch.where(region==areas[i],-areas_nums,region)
#                 areas_nums+=1
#             n_mask=torch.where(n_mask==i,region,n_mask)
#     n_mask=-n_mask

    return n_mask

def fast_cluster(feature,bandwidth=0.16):
    masks=[]
    areas=[]
    segments=[]

    for i in range(feature.shape[0]):
        n_mask=0
        n_feature=feature[i,...]
        label=torch.zeros(n_feature.shape[1],n_feature.shape[2]).cuda(cuda_id).float()
        check=0
        count=0
        while(torch.min(label)==0):
            candidate=torch.where(label==0,torch.tensor(1).float().cuda(cuda_id),torch.tensor(0).float().cuda(cuda_id)).nonzero()
            #print(len(candidate))
            seed=torch.randint(len(candidate),(1,))[0].long()
            mean=n_feature[:,candidate[seed][0].long(),candidate[seed][1].long()].view(n_feature.shape[0],1,1)
            mean=mean_shift(n_feature, mean, bandwidth)
            t_masks=get_mask(n_feature, mean, bandwidth)
            #print(len(candidate),n_mask)
            label=label+t_masks
            if n_mask==0:
                #r_masks=refine_mask(t_masks)
                n_masks=t_masks
                n_mask=torch.max(n_masks)
                
            else:
                #r_masks=refine_mask(t_masks)
                n_masks=fuse_mask(n_masks,t_masks)
                n_mask=torch.max(n_masks)
            #print(torch.max(n_masks))
            if len(candidate)==check:
                count+=1
            else:
                check=len(candidate)
            if count>10:
                bandwidth=bandwidth*1.1
                count=0
            if n_mask==70:
                bandwidth=bandwidth*1.1
            if n_mask==80:
                bandwidth=bandwidth*1.1  
            if n_mask==90:
                bandwidth=bandwidth*1.1
            if n_mask==100:
                bandwidth=bandwidth*1.1
            if n_mask>100:
                n_masks=fuse_mask(n_masks,torch.where(label==0,torch.tensor(1).float().cuda(cuda_id),torch.tensor(0).float().cuda(cuda_id)))
                break
    return n_masks

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    pad=nn.ReplicationPad2d(1)
    padding=0
    conv_mod = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                 padding=padding, bias=False)
    return nn.Sequential(pad,conv_mod)
   

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.gn1 = nn.GroupNorm(group_dim,planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.gn2 = nn.GroupNorm(group_dim,planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
            # print(residual.shape)
            # print(out.shape)
        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.gn1 = nn.GroupNorm(group_dim,planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.gn2 = nn.GroupNorm(group_dim,planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.gn3 = nn.GroupNorm(group_dim,planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.gn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)


        out += residual
        out = self.relu(out)

        return out
class rsn_cluster(nn.Module):


    def __init__(self, 
                 n_classes=64, 
                 block_config=[3, 16, 3, 3], 
                 input_size= (480, 640), 
                 version='scene'):

        super(rsn_cluster, self).__init__()
        self.inplanes = 64
        layers=[4, 10, 5, 5]
        block=BasicBlock
        # Encoder
        self.conv1=conv2DGroupNormRelu(3, 32, k_size=3,
                                                padding=1, stride=1, bias=False)
        self.conv2=conv2DGroupNormRelu(32, 64, k_size=3,
                                                padding=1, stride=1, bias=False)       
        self.layer1 = self._make_layer(block, 64, layers[0],stride=1)

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=1)
        # self.layer5 = conv2DGroupNormRelu(in_channels=128, k_size=3, n_filters=256,
        #                                         padding=1, stride=1, bias=False,group_dim=group_dim)


        # Pyramid Pooling Module
        #we need to modify the padding to keep the diminsion
        #remove 1 ,because the error of bn
        self.pyramid_pooling = pyramidPoolingGroupNorm(256, [[30,40],[12,16],[3,4],[1,1]],group_dim=group_dim)
        #self.global_pooling = globalPooling(256, 1)
        # Final conv layers
        #self.cbr_final = conv2DBatchNormRelu(512, 256, 3, 1, 1, False)
        #self.dropout = nn.Dropout2d(p=0.1, inplace=True)
        self.fuse0 = conv2DGroupNormRelu(in_channels=512, k_size=3, n_filters=256,
                                                padding=1, stride=1, bias=False,group_dim=group_dim)        
        self.fuse1 = conv2DGroupNormRelu(in_channels=256, k_size=3, n_filters=128,
                                                 padding=1, stride=1, bias=False,group_dim=group_dim)
        #we need to replace the upsampling unit with nearest and deconv2d
        self.deconv1 = deconv2DGroupNormRelu(in_channels=128, n_filters=128, k_size=4, 
                                                 stride=2, padding=1,output_padding=0, bias=False,group_dim=group_dim)
        self.fuse2 = conv2DGroupNormRelu(in_channels=256, k_size=3, n_filters=192,
                                                 padding=1, stride=1, bias=False,group_dim=group_dim)
        self.deconv2 = deconv2DGroupNormRelu(in_channels=192, n_filters=192, k_size=4, 
                                                 stride=2, padding=1,output_padding=0, bias=False,group_dim=group_dim)
        self.fuse3 = conv2DGroupNormRelu(in_channels=256, k_size=3, n_filters=256,
                                                 padding=1, stride=1, bias=False,group_dim=group_dim) 
        self.inplanes = 256
        self.regress1 = self._make_layer(block,128, 4, stride=1)

        self.regress2 = conv2DGroupNormRelu(in_channels=128, k_size=3, n_filters=64,
                                                  padding=1, stride=1, bias=False,group_dim=group_dim)
        self.regress3 = conv2DGroupNormRelu(in_channels=64, k_size=3, n_filters=32,
                                                 padding=1, stride=1, bias=False,group_dim=group_dim) 
        self.regress4 = conv2DRelu(in_channels=32, k_size=3, n_filters=1,
                                         padding=1, stride=1, bias=False)
        self.class0= conv2DGroupNormRelu(in_channels=258, k_size=1, n_filters=128,
                                                 padding=0, stride=1, bias=False,group_dim=group_dim)
        self.class1= conv2DGroupNormRelu(in_channels=128, k_size=3, n_filters=64,
                                                 padding=1, stride=1, bias=False,group_dim=group_dim)
        self.class2= conv2DRelu(in_channels=64, k_size=3, n_filters=64,
                                                 padding=1, stride=1, bias=False)
        self.class3= conv2DRelu(in_channels=64, k_size=3, n_filters=32,
                                                 padding=1, stride=1, bias=False)        
        self.class4= conv2D(in_channels=32, k_size=1, n_filters=16,
                                                 padding=0, stride=1, bias=False)
        self.outrefine1=conv2DGroupNormRelu(in_channels=514, k_size=1, n_filters=128,
                                                 padding=0, stride=1, bias=False,group_dim=group_dim)
        self.outrefine2=conv2DGroupNormRelu(in_channels=128, k_size=1, n_filters=64,
                                                 padding=0, stride=1, bias=False,group_dim=group_dim)
        self.outrefine3=conv2DRelu(in_channels=64, k_size=3, n_filters=32,
                                                 padding=1, stride=1, bias=False)
        self.outrefine4= conv2D(in_channels=32, k_size=1, n_filters=1,
                                                 padding=0, stride=1, bias=False)
        self.inrefine1=conv2DGroupNormRelu(in_channels=257, k_size=3, n_filters=128,
                                                 padding=1, stride=1, bias=False,group_dim=group_dim)
        self.inrefine2=conv2DGroupNormRelu(in_channels=128, k_size=3, n_filters=64,
                                                 padding=1, stride=1, bias=False,group_dim=group_dim)
        self.inrefine3=conv2DGroupNormRelu(in_channels=64, k_size=3, n_filters=32,
                                                 padding=1, stride=1, bias=False,group_dim=group_dim)
        self.inrefine4= conv2DRelu(in_channels=32, k_size=1, n_filters=16,
                                                 padding=0, stride=1, bias=False)
        self.inrefine5= conv2D(in_channels=16, k_size=1, n_filters=1,
                                                 padding=0, stride=1, bias=False)
        # self.reliable1=conv2DGroupNormRelu(in_channels=257, k_size=3, n_filters=128,
        #                                          padding=1, stride=1, bias=False,group_dim=group_dim)
        # self.reliable2=conv2DGroupNormRelu(in_channels=128, k_size=3, n_filters=64,
        #                                          padding=1, stride=1, bias=False,group_dim=group_dim)
        # self.reliable3= conv2DGroupNormRelu(in_channels=64, k_size=1, n_filters=32,
        #                                          padding=0, stride=1, bias=False,group_dim=group_dim)    
        # self.reliable4= conv2DGroupNormRelu(in_channels=32, k_size=1, n_filters=16,
        #                                          padding=0, stride=1, bias=False)
        # self.reliable5= conv2DRelu(in_channels=16, k_size=1, n_filters=1,
        #                                          padding=0, stride=1, bias=False)

        self.output=nn.ReLU(inplace=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.fill_(1)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.ReplicationPad2d(0),
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False,padding=0),
                nn.GroupNorm(group_dim,planes * block.expansion),
            )

        layers = []
        
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        #print(self.inplanes)
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    def forward(self, x,segments,flag,task):
        #print(x.shape)
        location_map=torch.cat([(torch.arange(x.shape[-1])/x.shape[-1]).unsqueeze(0).expand(x.shape[-2],x.shape[-1]).unsqueeze(0), \
            (torch.arange(x.shape[-2])/x.shape[-2]).unsqueeze(0).transpose(1,0).expand(x.shape[-2],x.shape[-1]).unsqueeze(0)],0).unsqueeze(0).float().cuda(cuda_id)
        #x=torch.cat([x,location_map],1)
        zero=torch.zeros(1).cuda(cuda_id)
        one=torch.ones(1).cuda(cuda_id)
        x = self.conv1(x)
        x=self.conv2(x)
        x1 = self.layer1(x)
        #half resolution
        x2 = self.layer2(x1)
        #print(x.shape)
        x = self.layer3(x2)
        #print(x.shape)
        x = self.layer4(x)
        #print(x.shape)
        # H, W -> H/2, W/2 
        x = self.pyramid_pooling(x)

        #x = self.cbr_final(x)
        #x = self.dropout(x)
        x = self.fuse0(x)
        x = self.fuse1(x)
        #print(x.shape)
        x = self.deconv1(x)
        #print(x.shape,x2.shape)
        x = self.fuse2(torch.cat((x,x2),1))
        x = self.deconv2(x)
        #print(x.shape)
        x_share = self.fuse3(torch.cat((x,x1),1))
        x=self.regress1(x_share)
        #print(x.shape)
        x=self.regress2(x)
        x=self.regress3(x)
        depth=self.regress4(x)

        # accurate_depth=depth
        # return depth,accurate_depth
        #clustering feature

        #accurate_depth=depth*reliable
        
        if flag==0:
            x_fuse=torch.cat([x_share,location_map],1)
            y=self.class0(x_fuse)
            y=self.class1(y)
            y=self.class2(y)
            y=self.class3(y)
            y=self.class4(y)
            with torch.no_grad():
                #masks=fast_cluster(y).view_as(depth)
                masks=segments.view_as(depth)
                #coarse depth
                coarse_depth=torch.zeros_like(depth).cuda(cuda_id)
                coarse_feature=torch.zeros_like(x_share).cuda(cuda_id)
                mean_features=torch.zeros(1,x_share.shape[1],torch.max(masks).long()+1).cuda(cuda_id)
                mean_depth=torch.zeros(torch.max(masks).long()+1).cuda(cuda_id)
                #print(torch.max(masks))
                for i in range(torch.min(masks).int(),torch.max(masks).int()+1):
                    index_r=torch.where(masks==i,one,zero)
                    mean_d=torch.sum(index_r*depth)/torch.sum(index_r)
                    mean_depth[i]=mean_d
                    coarse_depth=torch.where(masks==i,mean_d,coarse_depth)
                    mean_f=torch.sum((index_r*x_share).view(x_share.shape[0],x_share.shape[1],-1),dim=-1)/torch.sum(index_r)
                    #print(mean_f.shape,mean_features[...,i].shape)
                    mean_features[...,i]=mean_f
                    coarse_feature=torch.where(masks==i,mean_f.view(x_share.shape[0],x_share.shape[1],1,1),coarse_feature)

            #     #refine outer
            #     outer_feature=torch.zeros(1,2*x_share.shape[1]+2,torch.max(masks).long()+1,torch.max(masks).long()+1).cuda(cuda_id)
            #     for i in range(torch.min(masks).int(),torch.max(masks).int()+1):
            #         for j in range(torch.min(masks).int(),torch.max(masks).int()+1):
            #             if i!=j:
            #                 #print(outer_feature[...,i,j].shape,mean_depth[i].view(1,1).shape,mean_features[...,i].shape)
            #                 outer_feature[...,i,j]=torch.cat([mean_depth[i].view(1,1),mean_features[...,i],mean_depth[j].view(1,1),mean_features[...,j]],dim=-1)

            # outer=self.outrefine1(outer_feature)
            # outer=self.outrefine2(outer)
            # outer=self.outrefine3(outer)
            # outer_variance=self.outrefine4(outer)
            # outer_depth=torch.zeros(torch.max(masks).long()+1).cuda(cuda_id)
            # #mean_depth_map=coarse_depth+0
            # with torch.no_grad():
            #     for i in range(torch.min(masks).int(),torch.max(masks).int()+1):
            #         outer_depth[i]=(torch.sum(mean_depth*outer_variance[...,i,:])+mean_depth[i])/torch.sum(outer_variance[...,i,0]+1)
            #         #outer_depth[i]=(torch.sum(mean_depth*outer_variance[...,i,:])+mean_depth[i])
            #         coarse_depth=torch.where(masks==i,outer_depth[i],coarse_depth)+0
            #refine inner
            coarse_depth=self.output(coarse_depth)
            inner_feature= torch.cat([coarse_depth,x_share-coarse_feature],1)
            inner=self.inrefine1(inner_feature)
            inner=self.inrefine2(inner)
            inner=self.inrefine3(inner)
            inner=self.inrefine4(inner)
            inner_variance=self.inrefine5(inner)
            # #inner_variance[:,0,...]=inner_variance[:,0,...]/torch.max(inner_variance[:,0,...])
            # reliable_to_depth=(inner_variance[:,0,...]/torch.max(inner_variance[:,0,...])).unsqueeze(1)
            # variance_on_cosrse=inner_variance[:,1,...].unsqueeze(1)
            # #print(inner_variance.shape)
            # accurate_depth=depth*reliable_to_depth+(coarse_depth*variance_on_cosrse)*(1-reliable_to_depth)
            loss_var,loss_dis,loss_reg = cluster_loss(y,segments.long(),device_id=cuda_id)
            loss_var=loss_var.reshape((y.shape[0],1))
            loss_dis=loss_dis.reshape((y.shape[0],1))
            loss_reg=loss_reg.reshape((y.shape[0],1))
            accurate_depth=self.output(inner_variance+coarse_depth)
            print(torch.mean(depth).item(),torch.mean(coarse_depth).item())
            return depth,accurate_depth,loss_var,loss_dis,loss_reg
        else:
            if task=='train':
                with torch.no_grad():
                    masks=fast_cluster(y).view_as(depth)
                    print(torch.max(masks))

                loss_var,loss_dis,loss_reg = cluster_loss(y,segments.long())
                loss_var=loss_var.reshape((y.shape[0],1))
                loss_dis=loss_dis.reshape((y.shape[0],1))
                loss_reg=loss_reg.reshape((y.shape[0],1))
                return depth,masks,loss_var,loss_dis,loss_reg
            elif task=='test':

                loss_var,loss_dis,loss_reg = cluster_loss(y,segments.long())
                loss_var=loss_var.reshape((y.shape[0],1))
                loss_dis=loss_dis.reshape((y.shape[0],1))
                loss_reg=loss_reg.reshape((y.shape[0],1))
                return depth,loss_var,loss_dis,loss_reg
            elif task=='eval':

                x_fuse=torch.cat([x_share,location_map],1)
                masks=segments.view_as(depth)
                #coarse depth
                coarse_depth=depth+0
                coarse_feature=x_fuse+0
                mean_features=torch.zeros(1,x_fuse.shape[1],torch.max(masks).long()+1).cuda(cuda_id)
                mean_depth=torch.zeros(torch.max(masks).long()+1).cuda(cuda_id)
                
                for i in range(torch.min(masks).int(),torch.max(masks).int()+1):
                    index_r=torch.where(masks==i,one,zero)
                    mean_d=torch.sum(index_r*depth)/torch.sum(index_r)
                    mean_depth[i]=mean_d+0
                    coarse_depth=torch.where(masks==i,mean_depth[i],coarse_depth)
                    mean_f=torch.sum((index_r*x_fuse).view(x_fuse.shape[0],x_fuse.shape[1],-1),dim=-1)/torch.sum(index_r)
                    #print(mean_f.shape,mean_features[...,i].shape)
                    mean_features[...,i]=mean_f
                    coarse_feature=torch.where(masks==i,mean_f.view(x_fuse.shape[0],x_fuse.shape[1],1,1),coarse_feature)

                #refine outer
                # outer_feature=torch.zeros(1,2*x_fuse.shape[1]+2,torch.max(masks).long()-torch.min(masks).long()+1,torch.max(masks).long()-torch.min(masks).long()+1).cuda(cuda_id)
                # for i in range(torch.min(masks).int(),torch.max(masks).int()+1):
                #     for j in range(torch.min(masks).int(),torch.max(masks).int()+1):
                #         if i!=j:
                #             #print(outer_feature[...,i,j].shape,mean_depth[i].view(1,1).shape,mean_features[...,i].shape)
                #             outer_feature[...,i,j]=torch.cat([mean_depth[i].view(1,1),mean_features[...,i],mean_depth[j].view(1,1),mean_features[...,j]],dim=-1)

                # outer=self.outrefine1(outer_feature)
                # outer=self.outrefine2(outer)
                # outer=self.outrefine3(outer)
                # outer_variance=self.outrefine4(outer)
                # outer_depth=torch.zeros(torch.max(masks).long()-torch.min(masks).long()+1).cuda(cuda_id)
                # #mean_depth_map=coarse_depth+0
                # # print(torch.min(masks))
                # # print(torch.sum(torch.where(masks==0,torch.ones(1).cuda(cuda_id),torch.zeros(1).cuda(cuda_id))))
                # for i in range(torch.min(masks).int(),torch.max(masks).int()+1):
                #     outer_depth[i]=(torch.sum(mean_depth*outer_variance[...,i,:])+mean_depth[i])/(torch.sum(outer_variance[...,i,:])+1)
                #     #outer_depth[i]=(torch.sum(mean_depth*outer_variance[...,i,:])+mean_depth[i])
                #     coarse_depth=torch.where(masks==i,outer_depth[i],coarse_depth)+0
                #print(torch.max(coarse_depth),torch.mean(mean_depth),torch.mean(outer_depth),torch.max(outer_variance))
                #mean_depth_map=coarse_depth+0
                #refine inner
                inner_feature= torch.cat([coarse_depth,x_fuse-coarse_feature],1)

                #print('inner_feature',torch.max(inner_feature).item())
                inner=self.inrefine1(inner_feature)
                #print('inner_1',torch.max(inner).item())
                inner=self.inrefine2(inner)
                #print('inner_2',torch.max(inner).item())
                inner=self.inrefine3(inner)
                #print('inner_3',torch.max(inner).item())
                inner=self.inrefine4(inner)
                inner_variance=self.inrefine5(inner)
                accurate_depth=inner_variance

                # inner_feature= torch.cat([depth,x_share],1)
                # relialbe=self.reliable1(inner_feature)
                # relialbe=self.reliable2(relialbe)
                # relialbe=self.reliable3(relialbe)
                # relialbe=self.reliable4(relialbe)
                # relialbe=self.reliable5(relialbe)
                # accurate_depth=relialbe
                # print('inner_variance',torch.max(inner_variance).item())
                # inner_variance[:,0,...]=inner_variance[:,0,...]/torch.max(inner_variance[:,0,...])
                # reliable_to_depth=(torch.exp(-relialbe[:,0,...])).unsqueeze(1)
                # reliable_to_coarse=(torch.exp(-inner_variance[:,0,...])).unsqueeze(1)
                # variance_on_depth=relialbe[:,1,...].unsqueeze(1)
                # variance_on_cosrse=inner_variance[:,1,...].unsqueeze(1)
                # print('reliable_depth: %.2f reliable_coarse: %.2f variance_depth %.2f variance_coarse %.2f'%(torch.mean(reliable_to_depth).item(), \
                #                 torch.mean(reliable_to_coarse).item(),torch.mean(variance_on_depth).item(),torch.mean(variance_on_cosrse).item()))
                # #print('variance %.2f'%(torch.mean(inner_variance).item()))
                # relialbe_weights=reliable_to_coarse+reliable_to_depth
                # # #print(inner_variance.shape)
                # accurate_depth=(depth*variance_on_depth*reliable_to_coarse+coarse_depth*variance_on_cosrse*reliable_to_coarse)/ \
                #                                         (torch.where(relialbe_weights==0,torch.ones(1).cuda(cuda_id),relialbe_weights))
                # refined_depth=depth*variance_on_depth
                # coarse_depth=coarse_depth*variance_on_cosrse
                # accurate_depth=(coarse_depth*reliable_to_coarse+refined_depth*(1-reliable_to_coarse))
                # accurate_depth=refined_depth*reliable_to_depth
                # print('depth',torch.max(depth).item())
                # print('coarse',torch.max(coarse_depth).item())
                # print('accurate',torch.max(accurate_depth).item())
                # loss_var,loss_dis,loss_reg = cluster_loss(y,segments.long())
                # loss_var=loss_var.reshape((y.shape[0],1))
                # loss_dis=loss_dis.reshape((y.shape[0],1))
                # loss_reg=loss_reg.reshape((y.shape[0],1))

                # accurate_depth=inner_variance
                # simple refinement
                # x_fuse=x_share+depth.expand_as(x_share)
                # inner=self.inrefine1(x_fuse)
                # inner=self.inrefine2(inner)
                # inner=self.inrefine3(inner)
                # inner=self.inrefine4(inner)
                # accurate_depth=self.inrefine5(inner)
                accurate_depth=depth
                return depth,accurate_depth
        


