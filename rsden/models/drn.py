# -*- coding: utf-8 -*-
# @Author: lidong
# @Date:   2018-03-20 18:01:52
# @Last Modified by:   yulidong
# @Last Modified time: 2018-05-21 19:15:23

import torch
import numpy as np
import torch.nn as nn
import math
from math import ceil
from torch.autograd import Variable

from rsden import caffe_pb2
from rsden.models.utils import *
drn_specs = {
    'scene': 
    {
         'n_classes': 9,
         'input_size': (540, 960),
         'block_config': [3, 4, 23, 3],
    },

}
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
class drn(nn.Module):


    def __init__(self, 
                 n_classes=9, 
                 block_config=[3, 4, 6, 3], 
                 input_size= (480, 640), 
                 version='scene'):

        super(drn, self).__init__()
        self.inplanes = 64
        self.block_config = drn_specs[version]['block_config'] if version is not None else block_config
        self.n_classes = drn_specs[version]['n_classes'] if version is not None else n_classes
        self.input_size = drn_specs[version]['input_size'] if version is not None else input_size
        layers=[3, 4, 6, 3]
        block=BasicBlock
        # Encoder
        self.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.full1=conv2DBatchNormRelu(in_channels=64, k_size=3, n_filters=128,
                                                padding=2, stride=1, bias=False)
        self.full2=conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=128,
                                                padding=2, stride=1, bias=False) 

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # Pyramid Pooling Module
        #we need to modify the padding to keep the diminsion
        #remove 1 ,because the error of bn
        self.pyramid_unit = pyramidPooling(128, [[120,160],[60,80],[48,64],[24,32],[12,16],[6,8],[3,4]])

        self.infer1 = conv2DBatchNormRelu(in_channels=254, k_size=3, n_filters=128,
                                                padding=1, stride=1, bias=False)        
        self.infer2 = conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64,
                                                 padding=1, stride=1, bias=False)
        self.infer3 = conv2DBatchNormRelu(in_channels=64, k_size=3, n_filters=32,
                                                 padding=1, stride=1, bias=False)        
        self.infer4 = conv2DBatchNormRelu(in_channels=32, k_size=3, n_filters=16,
                                                 padding=1, stride=1, bias=False) 
        self.infer5 = conv2DRelu(in_channels=16, k_size=3, n_filters=1,
                                         padding=1, stride=1, bias=False) 
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)                                                                                                                 
    def forward(self, x):
        rgb=x[:,0:4,:,:]
        #print(x.shape)
        out=[]
        for i in range(3):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)


            x = self.layer1(x)
            x=self.full1(x)
            x=self.full2(x)

            # H, W -> H/2, W/2 
            x = self.pyramid_unit(x)

            #x = self.cbr_final(x)
            #x = self.dropout(x)
            x = self.infer1(x)
            #x=torch.cat((x,x1),1)
            x=self.infer2(x)
            x=self.infer3(x)
            x=self.infer4(x)
            x=self.infer5(x)
            #print(x.shape)
            #print(rgb.shape)
            out.append(x)
            x=torch.cat([rgb,x],1)

        return out


