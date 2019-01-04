# -*- coding: utf-8 -*-
# @Author: lidong
# @Date:   2018-03-20 18:01:52
# @Last Modified by:   yulidong
# @Last Modified time: 2019-01-01 21:59:28

import torch
import numpy as np
import torch.nn as nn
import math
from math import ceil
from torch.autograd import Variable
from rsden.cluster_loss import *
from rsden import caffe_pb2
from rsden.models.utils import *
import time
cuda_id=2
group_dim=32


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
class memory(nn.Module):


    def __init__(self, 
                 n_classes=64, 
                 block_config=[3, 16, 3, 3], 
                 input_size= (480, 640), 
                 version='scene'):

        super(memory, self).__init__()
        self.inplanes = 64
        layers=[2, 2, 2, 2,2]
        block=BasicBlock
        # Encoder
        self.conv1=conv2DGroupNormRelu(128, 64, k_size=3,
                                                padding=1, stride=2, bias=False)
        self.layer1 = self._make_layer(block, 128, layers[0],stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)
        self.layer5 = self._make_layer(block, 128, layers[4], stride=2)
        #3*5*c
        self.convert1=torch.nn.Linear(20*128,2048)
        self.convert2=torch.nn.Linear(2048,1024)
        self.output=nn.ReLU(inplace=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, math.sqrt(2. / n))


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
    def forward(self, x):
        zero=torch.zeros(1).cuda()
        one=torch.ones(1).cuda()
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        #print(x.shape)
        x=x.view(x.shape[0],-1)
        x=self.convert1(x)
        x=self.convert2(x)
        return x
        


