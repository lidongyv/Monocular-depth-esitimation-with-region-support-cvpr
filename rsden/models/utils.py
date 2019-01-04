import torch
import torch.nn as nn
import torch.nn.functional as F

group_dim=16
class conv2DBatchNorm(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True, dilation=1):
        super(conv2DBatchNorm, self).__init__()

        if dilation > 1:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                 padding=padding, stride=stride, bias=bias, dilation=dilation)

        else:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                 padding=padding, stride=stride, bias=bias, dilation=1)


        self.cb_unit = nn.Sequential(conv_mod,
                                     nn.BatchNorm2d(int(n_filters)),)

    def forward(self, inputs):
        outputs = self.cb_unit(inputs)
        return outputs
class conv2DGroupNorm(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True, dilation=1,group_dim=group_dim):
        super(conv2DGroupNorm, self).__init__()

        if dilation > 1:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                 padding=padding, stride=stride, bias=bias, dilation=dilation)

        else:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                 padding=padding, stride=stride, bias=bias, dilation=1)


        self.cb_unit = nn.Sequential(conv_mod,
                                     nn.GroupNorm(group_dim,int(n_filters)),)

    def forward(self, inputs):
        outputs = self.cb_unit(inputs)
        return outputs

class deconv2DBatchNorm(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True):
        super(deconv2DBatchNorm, self).__init__()

        self.dcb_unit = nn.Sequential(nn.ConvTranspose2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                               padding=padding, stride=stride, bias=bias),
                                 nn.BatchNorm2d(int(n_filters)),)

    def forward(self, inputs):
        outputs = self.dcb_unit(inputs)
        return outputs
class deconv2DGroupNorm(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True,group=group_dim):
        super(deconv2DGroupNorm, self).__init__()

        self.dcb_unit = nn.Sequential(nn.ConvTranspose2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                               padding=padding, stride=stride, bias=bias),
                                 nn.GroupNorm(group_dim,int(n_filters)),)

    def forward(self, inputs):
        outputs = self.dcb_unit(inputs)
        return outputs

class deconv2D(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True):
        super(deconv2D, self).__init__()

        self.dcb_unit = nn.Sequential(nn.ConvTranspose2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                               padding=padding, stride=stride, bias=bias),
                                 )

    def forward(self, inputs):
        outputs = self.dcb_unit(inputs)
        return outputs

class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding=0, bias=True, dilation=1):
        super(conv2DBatchNormRelu, self).__init__()

        if dilation > 1:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size, 
                                 padding=padding, stride=stride, bias=bias, dilation=dilation)

        else:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size, 
                                 padding=padding, stride=stride, bias=bias, dilation=1)

        self.cbr_unit = nn.Sequential(conv_mod,
                                      nn.BatchNorm2d(int(n_filters)),
                                      nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs
class conv2DGroupNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding=1, bias=True, dilation=1,group_dim=group_dim):
        super(conv2DGroupNormRelu, self).__init__()
        
        #padding=0
        if dilation > 1:
            pad=nn.ReplicationPad2d(dilation)
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size, 
                                 padding=0, stride=stride, bias=bias, dilation=dilation)

        else:
            pad=nn.ReplicationPad2d(padding)
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size, 
                                 padding=0, stride=stride, bias=bias, dilation=1)

        self.cbr_unit = nn.Sequential(pad
                                      ,conv_mod,
                                      nn.GroupNorm(group_dim,int(n_filters)),
                                      nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class conv2D(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding=1, bias=True, dilation=1,group_dim=group_dim):
        super(conv2D, self).__init__()
        pad=nn.ReplicationPad2d(padding)
        padding=0
        if dilation > 1:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size, 
                                 padding=padding, stride=stride, bias=bias, dilation=dilation)

        else:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size, 
                                 padding=padding, stride=stride, bias=bias, dilation=1)

        self.cbr_unit = nn.Sequential(pad,conv_mod)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs
class conv2DRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding=1, bias=True, dilation=1,group_dim=group_dim):
        super(conv2DRelu, self).__init__()
        pad=nn.ReplicationPad2d(padding)
        padding=0
        if dilation > 1:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size, 
                                 padding=padding, stride=stride, bias=bias, dilation=dilation)

        else:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size, 
                                 padding=padding, stride=stride, bias=bias, dilation=1)

        self.cbr_unit = nn.Sequential(pad,conv_mod,
                                      nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs
class deconv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride,output_padding=0, padding=0, bias=True):
        super(deconv2DBatchNormRelu, self).__init__()

        self.dcbr_unit = nn.Sequential(nn.ConvTranspose2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                                padding=padding, stride=stride, bias=bias,output_padding=output_padding),
                                 nn.BatchNorm2d(int(n_filters)),
                                 nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.dcbr_unit(inputs)
        return outputs
class uppooling(nn.Module):
    def __init__(self, in_channels, n_filters, k_size=3, stride=1,output_padding=0, padding=0, bias=False,group_dim=group_dim,check=False):
        super(uppooling, self).__init__()

        self.conva = nn.Sequential( nn.ReplicationPad2d(1),
                                    nn.Conv2d(int(in_channels), int(n_filters), kernel_size=(3,3),padding=0, stride=1, bias=bias)
                                   )
        self.convb = nn.Sequential( nn.ReplicationPad2d((1,1,1,0)),
                                    nn.Conv2d(int(in_channels), int(n_filters), kernel_size=(2,3),padding=0, stride=1, bias=bias)
                                   )
        self.convc = nn.Sequential( nn.ReplicationPad2d((1,0,1,1)),
                                    nn.Conv2d(int(in_channels), int(n_filters), kernel_size=(3,2),padding=0, stride=1, bias=bias)
                                   )
        self.convd = nn.Sequential( nn.ReplicationPad2d((1,0,1,0)),
                                    nn.Conv2d(int(in_channels), int(n_filters), kernel_size=(2,2),padding=0, stride=1, bias=bias)
                                   )
        self.norm=nn.GroupNorm(group_dim,int(n_filters))
        self.check=check
        self.relu=nn.ReLU(inplace=True)
        self.stride=stride
    def forward(self, x):
        outa=self.conva(x)
        outb=self.convb(x)
        outc=self.convc(x)
        outd=self.convd(x)
        #print(x.shape,outa.shape,outb.shape)
        outab=torch.reshape(torch.stack((outa,outb),dim=-1),(outa.shape[0],outa.shape[1],outa.shape[2],2*outa.shape[3]))
        outcd=torch.reshape(torch.stack((outc,outd),dim=-1),(outc.shape[0],outc.shape[1],outc.shape[2],2*outc.shape[3]))
        outputs=torch.reshape(torch.stack((outab,outcd),dim=-2),(outc.shape[0],outc.shape[1],2*outc.shape[2],2*outc.shape[3]))
        outputs=self.norm(outputs)
        if self.check==True:
            outputs=self.relu(outputs)
        return outputs
class upprojection(nn.Module):
    def __init__(self, in_channels, n_filters, k_size=3, stride=2,output_padding=0, padding=0, bias=True,group_dim=group_dim):
        super(upprojection, self).__init__()
        self.uppool1=uppooling(in_channels, n_filters,check=True)
        self.conv1=nn.Sequential( nn.ReplicationPad2d(1),
                                    nn.Conv2d(int(in_channels), int(n_filters), kernel_size=(3,3),padding=0, stride=1, bias=bias)
                                   )
        self.uppool2=uppooling(in_channels, n_filters,check=False)
        self.relu=nn.ReLU(inplace=True)
        self.stride=stride
    def forward(self, inputs):
        h, w = inputs.shape[-2:]
        #print(h,w)
        x=self.uppool1(inputs)
        x=self.conv1(x)
        y=self.uppool2(inputs)
        outputs=x+y
        outputs=self.relu(outputs)
        return outputs
class up2DGroupNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride=2,output_padding=0, padding=0, bias=True,group_dim=group_dim):
        super(up2DGroupNormRelu, self).__init__()

        self.dcbr_unit = nn.Sequential(conv2DGroupNormRelu(int(in_channels), int(n_filters), k_size=k_size,
                                                padding=padding, stride=1, bias=bias))
        self.stride=stride
    def forward(self, inputs):
        h, w = inputs.shape[-2:]
        #print(h,w)
        inputs=F.interpolate(inputs, size=(h*self.stride,w*self.stride), mode='bilinear',align_corners=False)
        outputs = self.dcbr_unit(inputs)
        return outputs
class deconv2DGroupNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride,output_padding=0, padding=0, bias=True,group_dim=group_dim):
        super(deconv2DGroupNormRelu, self).__init__()
        self.dcbr_unit = nn.Sequential( 
                                        nn.ConvTranspose2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                                padding=padding, stride=stride, bias=bias,output_padding=output_padding),
                                        # nn.ReplicationPad2d((1,0,1,0)),
                                        nn.GroupNorm(group_dim,int(n_filters)),
                                        nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.dcbr_unit(inputs)
        return outputs
class deconv2DRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride,output_padding=0, padding=0, bias=True):
        super(deconv2DRelu, self).__init__()

        self.dcbr_unit = nn.Sequential(nn.ConvTranspose2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                                padding=padding, stride=stride, bias=bias,output_padding=output_padding),
                                 
                                 nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.dcbr_unit(inputs)
        return outputs


class pyramidPoolingGroupNorm(nn.Module):

    def __init__(self, in_channels, pool_sizes,group_dim):
        super(pyramidPoolingGroupNorm, self).__init__()

        self.paths = []
        for i in range(len(pool_sizes)):
            if pool_sizes[i][0]>1:
                self.paths.append(conv2DGroupNormRelu(in_channels=in_channels, k_size=3, n_filters=int(in_channels / len(pool_sizes)),
                                                padding=1, stride=1, bias=False,group_dim=group_dim))
            else:
                self.paths.append(conv2DGroupNormRelu(in_channels=in_channels, k_size=1, n_filters=int(in_channels / len(pool_sizes)),
                                                padding=0, stride=1, bias=False,group_dim=group_dim))

        self.path_module_list = nn.ModuleList(self.paths)
        self.pool_sizes = pool_sizes

    def forward(self, x):
        output_slices = [x]
        h, w = x.shape[2:]
        #print(h,w)
        for module, pool_size in zip(self.path_module_list, self.pool_sizes):
            if pool_size[0]>1:
                out = F.adaptive_avg_pool2d(x, ((h//pool_size[0], w//pool_size[1])))
            else:
                out = F.adaptive_avg_pool2d(x, ((pool_size[0], pool_size[1])))
            #print(out.shape)
            #print(pool_size)
            out = module(out)
            out = F.interpolate(out, size=(h,w), mode='bilinear',align_corners=False)
            output_slices.append(out)

        return torch.cat(output_slices, dim=1)
class pyramidPoolingGroupNorm_constant(nn.Module):

    def __init__(self, in_channels, pool_sizes,group_dim):
        super(pyramidPoolingGroupNorm_constant, self).__init__()

        self.paths = []
        for i in range(len(pool_sizes)):
            self.paths.append(conv2DGroupNormRelu(in_channels=in_channels, k_size=3, n_filters=128,
                                                padding=1, stride=1, bias=False,group_dim=group_dim))

        self.path_module_list = nn.ModuleList(self.paths)
        self.pool_sizes = pool_sizes
        self.adjust=conv2DGroupNormRelu(in_channels=in_channels, k_size=1, n_filters=128,
                                                padding=0,dilation=1, stride=1, bias=False,group_dim=group_dim)
    def forward(self, x):
        output_slices = []
        h, w = x.shape[2:]
        adjust=self.adjust(x)
        output_slices.append(adjust)
        #print(h,w)
        for module, pool_size in zip(self.path_module_list, self.pool_sizes): 
            out = F.adaptive_avg_pool2d(x, ((pool_size[0], pool_size[1])))
            #print(pool_size)
            out = module(out)
            out = F.interpolate(out, size=(h,w), mode='bilinear',align_corners=False)
            output_slices.append(out)

        return torch.cat(output_slices, dim=1)
class AtrouspyramidPoolingGroupNorm(nn.Module):

    def __init__(self, in_channels, dilation,group_dim,kernel):
        super(AtrouspyramidPoolingGroupNorm, self).__init__()

        self.paths = []
        for i in range(len(dilation)):
            if i==0:
                self.paths.append(conv2DGroupNormRelu(in_channels=in_channels, k_size=kernel[i], n_filters=int(in_channels / len(dilation)),
                                    padding=dilation[i]-1,dilation=dilation[i], stride=1, bias=False,group_dim=group_dim))
            else:
                self.paths.append(conv2DGroupNormRelu(in_channels=in_channels, k_size=kernel[i], n_filters=int(in_channels / len(dilation)),
                                                padding=dilation[i],dilation=dilation[i], stride=1, bias=False,group_dim=group_dim))

        self.path_module_list = nn.ModuleList(self.paths)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             conv2DGroupNormRelu(in_channels=in_channels, k_size=1, n_filters=int(in_channels / len(dilation)),
                                                padding=0,dilation=1, stride=1, bias=False,group_dim=group_dim))
        self.dilation = dilation
        self._init_weight()
    def forward(self, x):
        output_slices = [x]
        h, w = x.shape[2:]
        out = self.global_avg_pool(x)
        out = F.interpolate(out, size=(h,w), mode='bilinear',align_corners=False)
        output_slices.append(out)
        #print(h,w)
        for module, pool_size in zip(self.path_module_list, self.dilation): 
            
            #print(pool_size)
            out = module(x)
            output_slices.append(out)

        return torch.cat(output_slices, dim=1)
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
class pyramidPooling(nn.Module):

    def __init__(self, in_channels, pool_sizes,group_dim):
        super(pyramidPooling, self).__init__()

        self.paths = []
        for i in range(len(pool_sizes)):
            self.paths.append(conv2DBatchNormRelu(in_channels=in_channels, k_size=3, n_filters=int(in_channels / len(pool_sizes)),
                                                padding=1, stride=1, bias=False))

        self.path_module_list = nn.ModuleList(self.paths)
        self.pool_sizes = pool_sizes

    def forward(self, x):
        output_slices = [x]
        h, w = x.shape[2:]
        #print(h,w)
        for module, pool_size in zip(self.path_module_list, self.pool_sizes): 
            out = F.adaptive_avg_pool2d(x, ((pool_size[0], pool_size[1])))
            #print(pool_size)
            out = module(out)
            out = F.interpolate(out, size=(h,w), mode='bilinear',align_corners=False)
            output_slices.append(out)

        return torch.cat(output_slices, dim=1)
class pyramidPooling_witoutbn(nn.Module):

    def __init__(self, in_channels, pool_sizes,group_dim):
        super(pyramidPooling_witoutbn, self).__init__()

        self.paths = []
        for i in range(len(pool_sizes)):
            self.paths.append(conv2DRelu(in_channels=in_channels, k_size=3, n_filters=int(in_channels / len(pool_sizes)),
                                                padding=1, stride=1, bias=False))

        self.path_module_list = nn.ModuleList(self.paths)
        self.pool_sizes = pool_sizes

    def forward(self, x):
        output_slices = [x]
        h, w = x.shape[2:]
        #print(h,w)
        for module, pool_size in zip(self.path_module_list, self.pool_sizes): 
            out = F.adaptive_avg_pool2d(x, ((pool_size[0], pool_size[1])))
            #print(pool_size)
            out = module(out)
            out = F.interpolate(out, size=(h,w), mode='bilinear',align_corners=False)
            output_slices.append(out)

        return torch.cat(output_slices, dim=1)

class globalPooling_withoutbn(nn.Module):

    def __init__(self, in_channels, pool_sizes):
        super(globalPooling_withoutbn, self).__init__()
        self.de1 = conv2DRelu(in_channels=32, k_size=5, n_filters=64,
                                                padding=1, stride=4, bias=False)
        self.de2 = conv2DRelu(in_channels=64, k_size=5, n_filters=128,
                                        padding=1, stride=4, bias=False)
        # self.de3 = conv2DBatchNormRelu(in_channels=1024, k_size=3, n_filters=2048,
        #                                 padding=1, stride=2, bias=False)
        # self.de4 = conv2DBatchNormRelu(in_channels=2048, k_size=3, n_filters=2048,
        #                                 padding=1, stride=2, bias=False)
        # self.de5 = conv2DBatchNormRelu(in_channels=2048, k_size=3, n_filters=2048,
        #                                 padding=1, stride=2, bias=False)
        # self.final1 = conv2DRelu(in_channels=2048, k_size=1, n_filters=1024,
        #                                 padding=0, stride=1, bias=False)
        self.final2 = conv2DRelu(in_channels=128, k_size=1, n_filters=64,
                                        padding=0, stride=1, bias=False)
        self.final3 = conv2DRelu(in_channels=64, k_size=1, n_filters=32,
                                        padding=0, stride=1, bias=False)
        # self.final4 = conv2DRelu(in_channels=256, k_size=1, n_filters=128,
        #                                 padding=0, stride=1, bias=False) 
        # self.final5 = conv2DRelu(in_channels=128, k_size=1, n_filters=64,
        #                                padding=0, stride=1, bias=False)
        #self.final6 = conv2D(in_channels=64, k_size=1, n_filters=32,
        #                                padding=0, stride=1, bias=False) 
        self.final7 = conv2D(in_channels=32, k_size=1, n_filters=1,
                                        padding=0, stride=1, bias=False)                                                                                                                                                                                                                                                                                                          
    def forward(self, x):
        h, w = x.shape[2:]
        x=self.de1(x)
        x=self.de2(x)
        # x=self.de3(x)
        # x=self.de4(x)
        # x=self.de5(x)
        #print(h,w)
        out = F.adaptive_avg_pool2d(x, ((1, 1)))
        #out=self.final1(out)
        out=self.final2(out)
        out=self.final3(out)
        #out=self.final4(out)
        #out=self.final5(out)
        #out=self.final6(out)
        out=self.final7(out)
        out = F.interpolate(out, size=(h,w), mode='bilinear',align_corners=False)


        return out

class globalPooling(nn.Module):

    def __init__(self, in_channels, pool_sizes):
        super(globalPooling, self).__init__()
        self.de1 = conv2DBatchNormRelu(in_channels=32, k_size=5, n_filters=64,
                                                padding=1, stride=4, bias=False)
        self.de2 = conv2DBatchNormRelu(in_channels=64, k_size=5, n_filters=128,
                                        padding=1, stride=4, bias=False)
        # self.de3 = conv2DBatchNormRelu(in_channels=1024, k_size=3, n_filters=2048,
        #                                 padding=1, stride=2, bias=False)
        # self.de4 = conv2DBatchNormRelu(in_channels=2048, k_size=3, n_filters=2048,
        #                                 padding=1, stride=2, bias=False)
        # self.de5 = conv2DBatchNormRelu(in_channels=2048, k_size=3, n_filters=2048,
        #                                 padding=1, stride=2, bias=False)
        # self.final1 = conv2DRelu(in_channels=2048, k_size=1, n_filters=1024,
        #                                 padding=0, stride=1, bias=False)
        self.final2 = conv2DRelu(in_channels=128, k_size=1, n_filters=64,
                                        padding=0, stride=1, bias=False)
        self.final3 = conv2DRelu(in_channels=64, k_size=1, n_filters=32,
                                        padding=0, stride=1, bias=False)
        # self.final4 = conv2DRelu(in_channels=256, k_size=1, n_filters=128,
        #                                 padding=0, stride=1, bias=False) 
        # self.final5 = conv2DRelu(in_channels=128, k_size=1, n_filters=64,
        #                                padding=0, stride=1, bias=False)
        #self.final6 = conv2D(in_channels=64, k_size=1, n_filters=32,
        #                                padding=0, stride=1, bias=False) 
        self.final7 = conv2D(in_channels=32, k_size=1, n_filters=1,
                                        padding=0, stride=1, bias=False)                                                                                                                                                                                                                                                                                                          
    def forward(self, x):
        h, w = x.shape[2:]
        x=self.de1(x)
        x=self.de2(x)
        # x=self.de3(x)
        # x=self.de4(x)
        # x=self.de5(x)
        #print(h,w)
        out = F.adaptive_avg_pool2d(x, ((1, 1)))
        #out=self.final1(out)
        out=self.final2(out)
        out=self.final3(out)
        #out=self.final4(out)
        #out=self.final5(out)
        #out=self.final6(out)
        out=self.final7(out)
        out = F.interpolate(out, size=(h,w), mode='bilinear',align_corners=False)


        return out

class bottleNeckPSP(nn.Module):
    
    def __init__(self, in_channels, mid_channels, out_channels, 
                 stride, dilation=1):
        super(bottleNeckPSP, self).__init__()
            
        self.cbr1 = conv2DBatchNormRelu(in_channels, mid_channels, 1, 1, 0, bias=False) 
        if dilation > 1: 
            self.cbr2 = conv2DBatchNormRelu(mid_channels, mid_channels, 3, 1, 
                                            padding=dilation, bias=False, 
                                            dilation=dilation) 
        else:
            self.cbr2 = conv2DBatchNormRelu(mid_channels, mid_channels, 3, 
                                            stride=stride, padding=1, 
                                            bias=False, dilation=1)
        self.cb3 = conv2DBatchNorm(mid_channels, out_channels, 1, 1, 0, bias=False)
        self.cb4 = conv2DBatchNorm(in_channels, out_channels, 1, stride, 0, bias=False)

    def forward(self, x):
        conv = self.cb3(self.cbr2(self.cbr1(x)))
        residual = self.cb4(x)
        return F.relu(conv+residual, inplace=True)


class bottleNeckIdentifyPSP(nn.Module):
    
    def __init__(self, in_channels, mid_channels, stride, dilation=1):
        super(bottleNeckIdentifyPSP, self).__init__()

        self.cbr1 = conv2DBatchNormRelu(in_channels, mid_channels, 1, 1, 0, bias=False) 
        if dilation > 1: 
            self.cbr2 = conv2DBatchNormRelu(mid_channels, mid_channels, 3, 1, 
                                            padding=dilation, bias=False, 
                                            dilation=dilation) 
        else:
            self.cbr2 = conv2DBatchNormRelu(mid_channels, mid_channels, 3, 
                                            stride=1, padding=1, 
                                            bias=False, dilation=1)
        self.cb3 = conv2DBatchNorm(mid_channels, in_channels, 1, 1, 0, bias=False)
        
    def forward(self, x):
        residual = x
        x = self.cb3(self.cbr2(self.cbr1(x)))
        return F.relu(x+residual, inplace=True)


