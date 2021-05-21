
import torch
import numpy as np
import torch.nn as nn
import os
import sys 
sys.path.append('/RAID5/Data/cwan/02-Project/01-AutoSegementation')

from model.resnet import resnet34
from torch.nn import functional as F
import matplotlib.pyplot as plt
from torch.nn import init

from utils.Breast import Breast
from torch.utils.data import DataLoader
up_kwargs = {'mode': 'bilinear', 'align_corners': True}



class CPFNet(nn.Module):
    def __init__(self, out_planes=1, ccm=True, norm_layer=nn.BatchNorm2d, is_training=True, expansion=2, base_channel=32):
        super(CPFNet, self).__init__()
        
        self.backbone = resnet34(pretrained=True)

        self.expansion = expansion
        self.base_channel = base_channel
        if self.expansion == 4 and self.base_channel == 64:
            expan = [512, 1024, 2048]
            spatial_ch = [128, 256]
        elif self.expansion == 4 and self.base_channel == 32:
            expan = [256, 512, 1024]
            spatial_ch = [32, 128]
            conv_channel_up = [256, 384, 512]
        elif self.expansion == 2 and self.base_channel == 32:
            expan = [128, 256, 512]
            spatial_ch = [64, 64]
            conv_channel_up = [128, 256, 512]
      
        conv_channel = expan[0]
        
        self.is_training = is_training

        self.sap = SAPblock(expan[-1])

        self.decoder5 = DecoderBlock(expan[-1], expan[-2], relu=False, last=True)  # 256
        self.decoder4 = DecoderBlock(expan[-2], expan[-3], relu=False)  # 128
        self.decoder3 = DecoderBlock(expan[-3], spatial_ch[-1], relu=False)  # 64
        self.decoder2 = DecoderBlock(spatial_ch[-1], spatial_ch[-2])  # 32

        self.mce_2 = GPG_2([spatial_ch[-1], expan[0], expan[1], expan[2]], width=spatial_ch[-1], up_kwargs=up_kwargs)
        self.mce_3 = GPG_3([expan[0], expan[1], expan[2]], width=expan[0], up_kwargs=up_kwargs)
        self.mce_4 = GPG_4([expan[1], expan[2]], width=expan[1], up_kwargs=up_kwargs)

        self.main_head = BaseNetHead(spatial_ch[0], out_planes, 2, is_aux=False, norm_layer=norm_layer)
       
        self.relu = nn.ReLU()

    def forward(self, x):# x[4,3,256,256]

        x = self.backbone.conv1(x)                 # x[4,3,256,256] --> x[4,64,128,128]
        x = self.backbone.bn1(x)                    
        c1 = self.backbone.relu(x)# 1/2  64
        
        x = self.backbone.maxpool(c1) #              x[4,64,128,128] --> x[4,64,64,64]
        c2 = self.backbone.layer1(x)# 1/4   64       x[4,64,64,64]   --> x[4,64,64,64]
        c3 = self.backbone.layer2(c2)# 1/8   128     x[4,64,64,64]   --> x[4,128,32,32]
        c4 = self.backbone.layer3(c3)# 1/16   256    x[4,128,32,32]  --> x[4,256,16,16]
        c5 = self.backbone.layer4(c4)# 1/32   512    x[4,256,16,16]  --> x[4,512,8,8]

        m2 = self.mce_2(c2,c3,c4,c5) # [4,64,64,64]
        m3 = self.mce_3(c3,c4,c5)    # [4,128,32,32]]
        m4 = self.mce_4(c4,c5)       # [4,256,16,16]

        c5=self.sap(c5) # [4,512,8,8]

        d4=self.relu(self.decoder5(c5)+m4)  #256 # self.decoder5(c5) [4,256,16,16]
        d3=self.relu(self.decoder4(d4)+m3)  #128 # self.decoder4(d4) [4,128,32,32]
        d2=self.relu(self.decoder3(d3)+m2)  #64  # self.decoder3(d3) [4,64,64,64]

        d1=self.decoder2(d2)+c1# 32 # d1[4,64,128,128]

        main_out = self.main_head(d1) # [4,out_planes,256,256]
        main_out = F.log_softmax(main_out, dim=1) # [4,out_planes,256,256]

        return main_out,main_out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)
                # m.weight.data.fill_(1)
                # m.bias.data.zero_()
        # return F.logsigmoid(main_out,dim=1)


class GPG_2(nn.Module):
    def __init__(self, in_channels, width=512, up_kwargs=None,norm_layer=nn.BatchNorm2d):
        super(GPG_2, self).__init__()
        self.up_kwargs = up_kwargs

        # expan = [128, 256, 512]
        # spatial_ch = [64, 64]
        # conv_channel_up = [128, 256, 512]
        # in_channels = [spatial_ch[-1], expan[0], expan[1], expan[2]]
        # [64, 128, 256, 512], 64

        self.conv5 = nn.Sequential( # c5:[4,512,8,8]
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),#(512,512,3,1)
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)) # out [4,64,8,8]

        self.conv4 = nn.Sequential( # c4:[4,256,16,16]
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),#(256,512,3,1)
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)) # out [4,64,16,16]

        self.conv3 = nn.Sequential( # c3:[4,128,32,32]
            nn.Conv2d(in_channels[-3], width, 3, padding=1, bias=False),#(128,512,3,1)
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)) # out [4,64,32,32]

        self.conv2 = nn.Sequential( # c2:[4,64,64,64]
            nn.Conv2d(in_channels[-4], width, 3, padding=1, bias=False),#(64,512,3,1)
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)) # out [4,64,64,64]

        self.conv_out = nn.Sequential(# [4,256,64,64]
            nn.Conv2d(4*width, width, 1, padding=0, bias=False),#(256,64,1,0)
            nn.BatchNorm2d(width))# [4,64,64,64]
        

        self.dilation1 = nn.Sequential(SeparableConv2d(4*width, width, kernel_size=3, padding=1, dilation=1, bias=False),#(256,64,3,1,1)
                                       nn.BatchNorm2d(width),
                                       nn.ReLU(inplace=True))

        self.dilation2 = nn.Sequential(SeparableConv2d(4*width, width, kernel_size=3, padding=2, dilation=2, bias=False),#(256,64,3,2,2)
                                       nn.BatchNorm2d(width),
                                       nn.ReLU(inplace=True))

        self.dilation3 = nn.Sequential(SeparableConv2d(4*width, width, kernel_size=3, padding=4, dilation=4, bias=False),#(256,64,3,4,4)
                                       nn.BatchNorm2d(width),
                                       nn.ReLU(inplace=True))

        self.dilation4 = nn.Sequential(SeparableConv2d(4*width, width, kernel_size=3, padding=8, dilation=8, bias=False),#(256,64,3,8,8)
                                       nn.BatchNorm2d(width),
                                       nn.ReLU(inplace=True))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)


    def forward(self, *inputs):
        # inputs[c2,c3,c4,c5]
        # c2:[4,64,64,64]
        # c3:[4,128,32,32]
        # c4:[4,256,16,16]
        # c5:[4,512,8,8]
        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]),self.conv3(inputs[-3]),self.conv2(inputs[-4])]# [c5,c4,c3,c2]
        _, _, h, w = feats[-1].size()# [4,256,64,64]

        feats[-2] = F.interpolate(feats[-2], (h, w), **self.up_kwargs)#
        feats[-3] = F.interpolate(feats[-3], (h, w), **self.up_kwargs)
        feats[-4] = F.interpolate(feats[-4], (h, w), **self.up_kwargs)

        feat = torch.cat(feats, dim=1)# [4,256,64,64]
        feat = torch.cat([self.dilation1(feat), self.dilation2(feat), self.dilation3(feat), self.dilation4(feat)], dim=1)
                         # [[4,64,64,64],[4,64,64,64],[4,64,64,64],[4,64,64,64]]
                         # [4,256,64,64]

        feat=self.conv_out(feat) # [4,64,64,64]

        return feat


class GPG_3(nn.Module):
    def __init__(self, in_channels, width=512, up_kwargs=None,norm_layer=nn.BatchNorm2d):
        super(GPG_3, self).__init__()
        self.up_kwargs = up_kwargs
        # expan = [128, 256, 512]
        # spatial_ch = [64, 64]
        # conv_channel_up = [128, 256, 512]
        # in_channels = [expan[0], expan[1], expan[2]]

        self.conv5 = nn.Sequential(# c5:[4,512,8,8]
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),# (512,512,3,1)
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))# out [4,512,8,8]

        self.conv4 = nn.Sequential(# c4:[4,256,16,16]
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),# (256,512,3,1)
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))# out [4,512,16,16]

        self.conv3 = nn.Sequential(# c3:[4,128,32,32]
            nn.Conv2d(in_channels[-3], width, 3, padding=1, bias=False),# (128,512,3,1)
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))# out [4,512,32,32]

        self.conv_out = nn.Sequential( # [4,512*3,32,32]
            nn.Conv2d(3*width, width, 1, padding=0, bias=False),
            nn.BatchNorm2d(width))# [4,512,32,32]
        
        # [4,512*3,32,32]
        self.dilation1 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=1, dilation=1, bias=False),# (512*3,512,3,1,1)
                                       nn.BatchNorm2d(width),
                                       nn.ReLU(inplace=True))

        self.dilation2 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=2, dilation=2, bias=False),# (512*3,512,3,2,2)
                                       nn.BatchNorm2d(width),
                                       nn.ReLU(inplace=True))

        self.dilation3 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=4, dilation=4, bias=False),# (512*3,512,3,4,4)
                                       nn.BatchNorm2d(width),
                                       nn.ReLU(inplace=True))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, *inputs):
        # inputs:[c3,c4,c5]
        # c3:[4,128,32,32]
        # c4:[4,256,16,16]
        # c5:[4,512,8,8]

        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3])]
        _, _, h, w = feats[-1].size()# [4,512,32,32]

        feats[-2] = F.interpolate(feats[-2], (h, w), **self.up_kwargs)
        feats[-3] = F.interpolate(feats[-3], (h, w), **self.up_kwargs)

        feat = torch.cat(feats, dim=1)# [4,512*3,32,32]
        feat = torch.cat([self.dilation1(feat), self.dilation2(feat), self.dilation3(feat)], dim=1)
                       # [[4,512,32,32],[4,512,32,32],[4,512,32,32]]
                       # [4,512*3,32,32]
        feat=self.conv_out(feat)# # [4,512,32,32]
        return feat


class GPG_4(nn.Module):
    def __init__(self, in_channels, width=512, up_kwargs=None,norm_layer=nn.BatchNorm2d):
        super(GPG_4, self).__init__()
        self.up_kwargs = up_kwargs
        # expan = [128, 256, 512]
        # spatial_ch = [64, 64]
        # conv_channel_up = [128, 256, 512]
        # in_channels = [expan[1], expan[2]]    

        self.conv5 = nn.Sequential(# c5:[4,512,8,8]
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))# out [4,256,8,8]

        self.conv4 = nn.Sequential(# c4:[4,256,16,16]
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))# out [4,256,16,16]

        self.conv_out = nn.Sequential(
            nn.Conv2d(2*width, width, 1, padding=0, bias=False),
            nn.BatchNorm2d(width))
        
        self.dilation1 = nn.Sequential(SeparableConv2d(2*width, width, kernel_size=3, padding=1, dilation=1, bias=False),# (1024,512,3,1,1)
                                       nn.BatchNorm2d(width),
                                       nn.ReLU(inplace=True))#

        self.dilation2 = nn.Sequential(SeparableConv2d(2*width, width, kernel_size=3, padding=2, dilation=2, bias=False),# (1024,512,3,2,2)
                                       nn.BatchNorm2d(width),
                                       nn.ReLU(inplace=True))#
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, *inputs):
        # inputs:[c4,c5]
        # c4:[4,256,16,16]
        # c5:[4,512,8,8]

        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2])]
        _, _, h, w = feats[-1].size() # [4,256,16,16]

        feats[-2] = F.interpolate(feats[-2], (h, w), **self.up_kwargs)

        feat = torch.cat(feats, dim=1)# [4,512,16,16]
        feat = torch.cat([self.dilation1(feat), self.dilation2(feat)], dim=1)
                       # [[4,256,16,16],[4,256,16,16]]
                       # [4,512,16,16]
        feat=self.conv_out(feat)# [4,256,16,16]
        return feat


class SeparableConv2d(nn.Module):
    # (2048,512,3,1,1)
    # (2048,512,3,2,2)
    # (2048,512,3,4,4)
    # (2048,512,3,8,8)
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, BatchNorm=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()
        # input [4,2048,64,64]
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation, groups=inplanes, bias=bias) # out [4,2048,64,64]
        self.bn = BatchNorm(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)# out [4,512,64,64]

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class SAPblock(nn.Module):
    def __init__(self, in_channels):
        # in_channels = 512

        super(SAPblock, self).__init__()

        self.conv3x3 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,dilation=1,kernel_size=3, padding=1)
        
        self.bn = nn.ModuleList([nn.BatchNorm2d(in_channels),nn.BatchNorm2d(in_channels),nn.BatchNorm2d(in_channels)]) 

        self.conv1x1 = nn.ModuleList([nn.Conv2d(in_channels=2*in_channels, out_channels=in_channels,dilation=1,kernel_size=1, padding=0),
                                      nn.Conv2d(in_channels=2*in_channels, out_channels=in_channels,dilation=1,kernel_size=1, padding=0)])

        self.conv3x3_1 = nn.ModuleList([nn.Conv2d(in_channels=in_channels, out_channels=in_channels//2,dilation=1,kernel_size=3, padding=1),
                                        nn.Conv2d(in_channels=in_channels, out_channels=in_channels//2,dilation=1,kernel_size=3, padding=1)])

        self.conv3x3_2 = nn.ModuleList([nn.Conv2d(in_channels=in_channels//2, out_channels=2,dilation=1,kernel_size=3, padding=1),
                                        nn.Conv2d(in_channels=in_channels//2, out_channels=2,dilation=1,kernel_size=3, padding=1)])

        self.conv_last = ConvBnRelu(in_planes=in_channels,out_planes=in_channels,ksize=1,stride=1,pad=0,dilation=1)



        self.gamma = nn.Parameter(torch.zeros(1))
    
        self.relu=nn.ReLU(inplace=True)

    def forward(self, x):# x [4,512,8,8]

        x_size= x.size()

        branches_1=self.conv3x3(x)# [4,512,8,8]
        branches_1=self.bn[0](branches_1) # [4,512,8,8]

        branches_2=F.conv2d(x,self.conv3x3.weight,padding=2,dilation=2)#share weight # [4,512,8,8]
        branches_2=self.bn[1](branches_2)

        branches_3=F.conv2d(x,self.conv3x3.weight,padding=4,dilation=4)#share weight # [4,512,8,8]
        branches_3=self.bn[2](branches_3)

        feat=torch.cat([branches_1,branches_2],dim=1) # [4,1024,8,8]

        # feat=feat_cat.detach()
        feat=self.relu(self.conv1x1[0](feat)) # [4,512,8,8]
        feat=self.relu(self.conv3x3_1[0](feat)) # [4,256,8,8]

        att=self.conv3x3_2[0](feat) # [4,2,8,8]
        att = F.softmax(att, dim=1) # [4,2,8,8]
        
        att_1=att[:,0,:,:].unsqueeze(1) # [4,1,8,8]
        att_2=att[:,1,:,:].unsqueeze(1) # [4,1,8,8]

        fusion_1_2=att_1*branches_1+att_2*branches_2 # [4,512,8,8]

        feat1=torch.cat([fusion_1_2,branches_3],dim=1) # [4,1024,8,8]
        # feat=feat_cat.detach()
        
        feat1=self.relu(self.conv1x1[0](feat1)) # [4,512,8,8]
        feat1=self.relu(self.conv3x3_1[0](feat1)) # [4,256,8,8]
        att1=self.conv3x3_2[0](feat1) # [4,2,8,8]
        att1 = F.softmax(att1, dim=1) # [4,2,8,8]
        
        att_1_2=att1[:,0,:,:].unsqueeze(1) # [4,1,8,8]
        att_3=att1[:,1,:,:].unsqueeze(1) # [4,1,8,8]

        ax=self.relu(self.gamma*(att_1_2*fusion_1_2+att_3*branches_3)+(1-self.gamma)*x) # gamma*[4,512,8,8] + (1-gamma)*[4,512,8,8] = [4,512,8,8]
        ax=self.conv_last(ax) # [4,512,8,8]

        return ax # [4,512,8,8]


class DecoderBlock(nn.Module):
    #d5 (512,256)
    #d4 (256,128)
    #d3 (128,64)
    #d2 (64,64)
    def __init__(self, in_planes, out_planes,norm_layer=nn.BatchNorm2d,scale=2,relu=True,last=False):
        super(DecoderBlock, self).__init__()
        # expan = [128, 256, 512]
        # spatial_ch = [64, 64]
        # conv_channel_up = [128, 256, 512]

        self.conv_3x3 = ConvBnRelu(in_planes, in_planes, 3, 1, 1, has_bn=True, norm_layer=norm_layer, has_relu=True, has_bias=False)#(in,in,3,1,1)

        self.conv_1x1 = ConvBnRelu(in_planes, out_planes, 1, 1, 0, has_bn=True, norm_layer=norm_layer, has_relu=True, has_bias=False)#(in,in//2,1,1,0)
       
        self.sap=SAPblock(in_planes)
        self.scale=scale
        self.last=last

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, x):

        if self.last==False:
            x = self.conv_3x3(x)
            # x=self.sap(x)

        if self.scale>1:
            x=F.interpolate(x,scale_factor=self.scale,mode='bilinear',align_corners=True)

        x=self.conv_1x1(x)
        return x

    
class ConvBnRelu(nn.Module):#(in,in,3,1,1)#(in,in//2,1,1,0)                     
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1, groups=1, has_bn=True, norm_layer=nn.BatchNorm2d,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize, stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)#[4,512,8,8]
                                                                              #[4,256,8,8]
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = nn.BatchNorm2d(out_planes)

        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):

        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x


class BaseNetHead(nn.Module):#[4,64,128,128]
    def __init__(self, in_planes, out_planes, scale, is_aux=False, norm_layer=nn.BatchNorm2d):
        super(BaseNetHead, self).__init__()
        # expan = [128, 256, 512]
        # spatial_ch = [64, 64]
        # conv_channel_up = [128, 256, 512]
        # BaseNetHead(64, 1, 2, is_aux=False, norm_layer=norm_layer)

        if is_aux:
            self.conv_1x1_3x3=nn.Sequential(
                ConvBnRelu(in_planes, 64, 1, 1, 0,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False),
                ConvBnRelu(64, 64, 3, 1, 1,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False))
        else:
            self.conv_1x1_3x3=nn.Sequential(
                ConvBnRelu(in_planes, 32, 1, 1, 0,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False),
                ConvBnRelu(32, 32, 3, 1, 1,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False))
        # self.dropout = nn.Dropout(0.1)
        if is_aux:
            self.conv_1x1_2 = nn.Conv2d(64, out_planes, kernel_size=1,
                                      stride=1, padding=0)
        else:
            self.conv_1x1_2 = nn.Conv2d(32, out_planes, kernel_size=1,
                                      stride=1, padding=0)
        self.scale = scale
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, x):

        if self.scale > 1:
            x = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=True) # [4,64,256,256]
        fm = self.conv_1x1_3x3(x) # [4,32,256,256]
        # fm = self.dropout(fm)
        output = self.conv_1x1_2(fm) # [4,2,256,256]
        return output

    
class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        in_size = inputs.size()
        inputs = inputs.view((in_size[0], in_size[1], -1)).mean(dim=2)
        inputs = inputs.view(in_size[0], in_size[1], 1, 1)

        return inputs

    


if __name__ == '__main__':

    data = Breast('data/train', (384, 384),mode='tr')

    dataloader_test = DataLoader(
        data,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    model = CPFNet(2)
    for i, (data, labels) in enumerate(dataloader_test):
        a,predict = model(data)
        print(data.shape)
        print(labels.shape)
        print((a.shape))
        break

    predict=torch.argmax(torch.exp(predict),dim=1)
    print(predict.shape==labels.shape)
    predict = predict.detach().numpy()



    imgl = np.array(data[0])
    la = np.array(labels[0])

    print(imgl.shape)
    print(la.shape)


#     plt.figure(figsize=(15,7))
#     plt.subplot(1,3,1)
#     plt.imshow(imgl,'gray')
#     plt.subplot(1, 3, 2)
#     plt.imshow(labels[0], 'gray')
#     plt.subplot(1,3,3)
#     plt.imshow(predict[0],'gray')
#
#     plt.show()



