# -*- coding: utf-8 -*-
# @Author: lidong
# @Date:   2018-03-20 18:01:52
# @Last Modified by:   yulidong
# @Last Modified time: 2018-08-22 10:02:06

import torch
import numpy as np
import torch.nn as nn
import math
from math import ceil
from torch.autograd import Variable
from rsden.cluster_loss import *
from rsden import caffe_pb2
from rsden.models.utils import *
rsn_specs = {
    'scene': 
    {
         'n_classes': 9,
         'input_size': (540, 960),
         'block_config': [3, 4, 23, 3],
    },

}
group_dim=4
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""

    if stride==1:
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
    if stride==2:
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=2, bias=False)       

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
                 block_config=[3, 4, 6, 3], 
                 input_size= (480, 640), 
                 version='scene'):

        super(rsn_cluster, self).__init__()
        self.inplanes = 64
        self.block_config = rsn_specs[version]['block_config'] if version is not None else block_config
        self.n_classes = rsn_specs[version]['n_classes'] if version is not None else n_classes
        self.input_size = rsn_specs[version]['input_size'] if version is not None else input_size
        layers=[3, 4, 6, 3]
        block=BasicBlock
        # Encoder
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=5,
                               bias=False,dilation=2)
        self.gn1 = nn.GroupNorm(group_dim,64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.full1=conv2DGroupNormRelu(in_channels=64, k_size=3, n_filters=64,
                                                padding=2, stride=1, bias=False,dilation=2,group_dim=group_dim)
        self.full2=conv2DGroupNormRelu(in_channels=64, k_size=3, n_filters=64,
                                                padding=3, stride=1, bias=False,dilation=2,group_dim=group_dim) 
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=1)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.layer4 = conv2DGroupNormRelu(in_channels=128, k_size=3, n_filters=256,
                                                padding=1, stride=1, bias=False,group_dim=group_dim)


        # Pyramid Pooling Module
        #we need to modify the padding to keep the diminsion
        #remove 1 ,because the error of bn
        self.pyramid_pooling = pyramidPoolingGroupNorm(256, [[120,160],[60,80],[48,64],[30,40],[24,32],[12,16],[6,8],[3,4]],group_dim=group_dim)
        #self.global_pooling = globalPooling(256, 1)
        # Final conv layers
        #self.cbr_final = conv2DBatchNormRelu(512, 256, 3, 1, 1, False)
        #self.dropout = nn.Dropout2d(p=0.1, inplace=True)
        self.deconv0 = conv2DGroupNormRelu(in_channels=512, k_size=3, n_filters=256,
                                                padding=1, stride=1, bias=False,group_dim=group_dim)        
        self.deconv1 = conv2DGroupNormRelu(in_channels=256, k_size=3, n_filters=128,
                                                 padding=1, stride=1, bias=False,group_dim=group_dim)
        self.deconv2 = up2DGroupNormRelu(in_channels=128, n_filters=128, k_size=3, 
                                                 stride=1, padding=1, bias=False,group_dim=group_dim)
        self.regress1 = conv2DGroupNormRelu(in_channels=128, k_size=3, n_filters=128,
                                                 padding=1, stride=1, bias=False,group_dim=group_dim)
        # self.regress2 = conv2DGroupNormRelu(in_channels=192, k_size=3, n_filters=128,
        #                                           padding=1, stride=1, bias=False)
        # self.regress3 = conv2DGroupNormRelu(in_channels=128, k_size=3, n_filters=64,
        #                                          padding=1, stride=1, bias=False)
        # self.regress4 = conv2DRelu(in_channels=64, k_size=3, n_filters=32,
        #                                          padding=1, stride=1, bias=False)        
        # self.final = conv2DRelu(in_channels=32, k_size=3, n_filters=16,
        #                                          padding=1, stride=1, bias=False) 
        # self.final2 = conv2DRelu(in_channels=16, k_size=3, n_filters=1,
        #                                  padding=1, stride=1, bias=False) 
        self.class1= conv2DGroupNormRelu(in_channels=192, k_size=3, n_filters=128,
                                                 padding=1, stride=1, bias=False,group_dim=group_dim)
        self.class2= conv2DGroupNormRelu(in_channels=128, k_size=3, n_filters=64,
                                                 padding=1, stride=1, bias=False,group_dim=group_dim)
        self.class3= conv2DGroupNormRelu(in_channels=64, k_size=3, n_filters=32,
                                                 padding=1, stride=1, bias=False,group_dim=group_dim)        
        self.class4= conv2D(in_channels=32, k_size=1, n_filters=16,
                                                 padding=0, stride=1, bias=False,group_dim=16)
        # self.class5= conv2DGroupNorm(in_channels=16, k_size=1, n_filters=8,
        #                                          padding=0, stride=1, bias=False,group_dim=8)
        # self.class5=nn.GroupNorm(1,16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False,padding=1),
                nn.GroupNorm(group_dim,planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)                                                                                                                 
    def forward(self, x,segments):
        inp_shape = x.shape[2:]
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu(x)


        x = self.layer1(x)
        x=self.full1(x)
        x1=self.full2(x)
        #print(x1.shape)
        x = self.layer2(x)
        #print(x.shape)
        #x = self.layer3(x)
        #print(x.shape)
        x = self.layer4(x)
        #print(x.shape)
        # H, W -> H/2, W/2 
        x = self.pyramid_pooling(x)

        #x = self.cbr_final(x)
        #x = self.dropout(x)
        x = self.deconv0(x)
     
        x = self.deconv1(x)
       
        x = self.deconv2(x)
       

        #128+128
        x=self.regress1(x)
        #print(x.shape)
        #print(x1.shape)
        x_f=torch.cat((x,x1),1)
        # x=self.regress2(x_f)
        # x=self.regress3(x)
        # x=self.regress4(x)
        # x=self.final(x)
        # x=self.final2(x)
        y=self.class1(x_f)
        y=self.class2(y)
        y=self.class3(y)
        y=self.class4(y)
        #y=self.class5(y)
        loss_var,loss_dis,loss_reg = cluster_loss(y,segments)
        loss_var=loss_var.reshape((y.shape[0],1))
        loss_dis=loss_dis.reshape((y.shape[0],1))
        loss_reg=loss_reg.reshape((y.shape[0],1))


        return y,loss_var,loss_dis,loss_reg
        #return x


