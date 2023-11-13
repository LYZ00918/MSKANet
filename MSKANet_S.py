import torch
import torch.nn as nn
import torch.nn.functional as F
from toolbox.backbone.mobilenet.MobileNetv2 import mobilenet_v2
'''
需求：通过建立一个模块，可以提取上下文信息
'''

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x1 = self.conv1(x)
        return self.sigmoid(x1)

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
class CMF(nn.Module):
    def __init__(self,in_channel,ratio):
        super(CMF,self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(3*in_channel//2,ratio)
        self.conv = nn.Conv2d(in_channel//2,in_channel,kernel_size=3,padding=1)
    def forward(self,x1_rgb,x1_d):
        x1_add = self.sa(x1_rgb + x1_d) * (x1_rgb + x1_d)
        x1_mul = self.sa(x1_rgb * x1_d) * (x1_rgb * x1_d)
        x1_sub = self.sa(x1_rgb - x1_d) * (x1_rgb - x1_d)
        b,c,h,w = x1_rgb.shape
        fuse = torch.cat((x1_sub,x1_mul,x1_add),dim=1)
        weight = self.ca(fuse)

        fuse1 = x1_add * weight[:,:c,:,:]
        fuse2 = x1_mul * weight[:,c:2*c,:,:]
        fuse3 = x1_sub * weight[:,2*c:,:,:]
        fuse_all = fuse1 + fuse2 + fuse3
        fuse_all = self.conv(fuse_all)

        return fuse_all

class HLG(nn.Module):
    def __init__(self,in_channel,out_channel,ratio):
        super(HLG,self).__init__()
        self.rgb = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )
        self.dep = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )
        self.rgb_branch1 = nn.Conv2d(out_channel, out_channel//2, 3, padding=1, dilation=1)
        self.rgb_branch2 = nn.Conv2d(out_channel, out_channel//2, 3, padding=3, dilation=3)
        self.rgb_branch3 = nn.Conv2d(out_channel, out_channel//2, 3, padding=5, dilation=5)
        self.rgb_branch4 = nn.Conv2d(out_channel, out_channel//2, 3, padding=7, dilation=7)

        self.d_branch1 = nn.Conv2d(out_channel, out_channel//2, 3, padding=1, dilation=1)
        self.d_branch2 = nn.Conv2d(out_channel, out_channel//2, 3, padding=3, dilation=3)
        self.d_branch3 = nn.Conv2d(out_channel, out_channel//2, 3, padding=5, dilation=5)
        self.d_branch4 = nn.Conv2d(out_channel, out_channel//2, 3, padding=7, dilation=7)

        self.CMF1 = CMF(out_channel,ratio)
        self.CMF2 = CMF(out_channel,ratio)
        self.CMF3 = CMF(out_channel,ratio)
        self.CMF4 = CMF(out_channel,ratio)

        self.conv = nn.Sequential(
            nn.Conv2d(out_channel*4, out_channel,kernel_size=1),
            nn.Conv2d(out_channel,out_channel,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )
        # self.conv_rgb = nn.Conv2d(out_channel,out_channel,kernel_size=3,padding=1)
    def forward(self,rgb,dep):
        rgb = self.rgb(rgb)
        dep = self.dep(dep)

        x1_rgb = self.rgb_branch1(rgb)
        x2_rgb = self.rgb_branch2(rgb)
        x3_rgb = self.rgb_branch3(rgb)
        x4_rgb = self.rgb_branch4(rgb)

        x1_d = self.d_branch1(dep)
        x2_d = self.d_branch2(dep)
        x3_d = self.d_branch3(dep)
        x4_d = self.d_branch4(dep)

        e111 = self.CMF1(x1_rgb,x1_d)
        e222 = self.CMF2(x2_rgb,x2_d)
        e333 = self.CMF3(x3_rgb,x3_d)
        e444 = self.CMF4(x4_rgb,x4_d)

        e_fuse = torch.cat((e111, e222, e333, e444), dim=1)
        # e_fuse = self.conv(e_fuse) + self.conv_rgb(rgb)
        e_fuse = self.conv(e_fuse)

        return e_fuse

class DecoderBlock(nn.Module):
    def __init__(self,in_channels, n_filters):
        super(DecoderBlock, self).__init__()
        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        # B, C/4, H, W -> B, C/4, H, W
        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3,
                                          stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU(inplace=True)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class SFAFMA_S(nn.Module):
    def __init__(self):
        super(SFAFMA_S, self).__init__()
        self.mobilenetv2_rgb = mobilenet_v2()
        self.mobilenetv2_dep = mobilenet_v2()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False)

        ##########   ENCODER    ###########
        self.HLG0 = HLG(16,16,ratio=2)
        self.HLG1 = HLG(24,24,ratio=4)
        self.HLG2 = HLG(32,32,ratio=4)
        self.HLG3 = HLG(160,60,ratio=16)
        self.HLG4 = HLG(320,100,ratio=16)



       ##########   DECODER    ###########
        self.decoder4 = DecoderBlock(100, 60)
        self.decoder3 = DecoderBlock(60, 32)
        self.decoder2 = DecoderBlock(32, 24)
        self.decoder1 = DecoderBlock(24, 16)
        self.decoder0 = DecoderBlock(16, 10)

        self.S4 = nn.Conv2d(60, 6, 3, stride=1, padding=1)
        self.S3 = nn.Conv2d(32, 6, 3, stride=1, padding=1)
        self.S2 = nn.Conv2d(24, 6, 3, stride=1, padding=1)
        self.S1 = nn.Conv2d(16, 6, 3, stride=1, padding=1)
        self.S0 = nn.Conv2d(10, 6, 3, stride=1, padding=1)

    def forward(self, rgb,dep):
        x0 = self.mobilenetv2_rgb.features[0:2](rgb)
        x1 = self.mobilenetv2_rgb.features[2:4](x0)
        x2 = self.mobilenetv2_rgb.features[4:7](x1)
        x3 = self.mobilenetv2_rgb.features[7:17](x2)
        x4 = self.mobilenetv2_rgb.features[17:18](x3)

        x_d = self.conv1(dep)
        d0 = self.mobilenetv2_dep.features[1:2](x_d)
        d1 = self.mobilenetv2_dep.features[2:4](d0)
        d2 = self.mobilenetv2_dep.features[4:7](d1)
        d3 = self.mobilenetv2_dep.features[7:17](d2)
        d4 = self.mobilenetv2_dep.features[17:18](d3)


        E0 = self.HLG0(x0, d0)
        E1 = self.HLG1(x1, d1)
        E2 = self.HLG2(x2, d2)
        E3 = self.HLG3(x3, d3)
        E4 = self.HLG4(x4, d4)
        # print(E0.shape)
        # print(E1.shape)
        # print(E2.shape)
        # print(E3.shape)
        # print(E4.shape)
        ##########   DECODER    ###########
        d4 = self.decoder4(E4) + E3
        z4 = self.S4(d4)

        d3 = self.decoder3(d4) + E2
        z3 = self.S3(d3)

        d2 = self.decoder2(d3) + E1
        z2 = self.S2(d2)

        d1 = self.decoder1(d2) + E0
        z1 = self.S1(d1)

        d0 = self.decoder0(d1)
        z0 = self.S0(d0)
        # fuse1 = self.finaldeconv1(d1)
        # fuse2 = self.finalrelu1(fuse1)
        # t0 = self.finalconv2(fuse2)

        # t0,t1,t2,t3,t4 = self.decoder(E0,E1,E2,E3,E4)

        # Z0 = F.interpolate(t0,size=(256,256),mode='bilinear')
        # Z1 = F.interpolate(t1,size=(256,256),mode='bilinear')
        # Z2 = F.interpolate(t2,size=(256,256),mode='bilinear')
        # Z3 = F.interpolate(t3,size=(256,256),mode='bilinear')
        # Z4 = F.interpolate(t4,size=(256,256),mode='bilinear')
        # t5 = F.interpolate(t5,size=(256,256),mode='bilinear')

        # return z0,z1,z2,z3,z4
        return z0,z1,z2,z3,z4,d0, d1, d2, d3, d4, E0, E1, E2, E3, E4

# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
#
#         self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()
#     def forward(self, x):
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = max_out
#         x1 = self.conv1(x)
#         return self.sigmoid(x1)
#
# class BasicConv2d(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
#         super(BasicConv2d, self).__init__()
#         self.conv = nn.Conv2d(in_planes, out_planes,
#                               kernel_size=kernel_size, stride=stride,
#                               padding=padding, dilation=dilation, bias=False)
#         self.bn = nn.BatchNorm2d(out_planes)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         return x
#
# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         return self.sigmoid(out)
# class CMF(nn.Module):
#     def __init__(self,in_channel,ratio):
#         super(CMF,self).__init__()
#         self.sa = SpatialAttention()
#         self.ca = ChannelAttention(3*in_channel//2,ratio)
#         self.conv = nn.Conv2d(in_channel//2,in_channel,kernel_size=3,padding=1)
#     def forward(self,x1_rgb,x1_d):
#         x1_add = self.sa(x1_rgb + x1_d) * (x1_rgb + x1_d)
#         x1_mul = self.sa(x1_rgb * x1_d) * (x1_rgb * x1_d)
#         x1_sub = self.sa(x1_rgb - x1_d) * (x1_rgb - x1_d)
#         b,c,h,w = x1_rgb.shape
#         fuse = torch.cat((x1_sub,x1_mul,x1_add),dim=1)
#         weight = self.ca(fuse)
#
#         fuse1 = x1_add * weight[:,:c,:,:]
#         fuse2 = x1_mul * weight[:,c:2*c,:,:]
#         fuse3 = x1_sub * weight[:,2*c:,:,:]
#         fuse_all = fuse1 + fuse2 + fuse3
#         fuse_all = self.conv(fuse_all)
#
#         return fuse_all
#
# class HLG(nn.Module):
#     def __init__(self,in_channel,out_channel,ratio):
#         super(HLG,self).__init__()
#         self.rgb = nn.Sequential(
#             nn.Conv2d(in_channel,out_channel,kernel_size=3,padding=1),
#             nn.BatchNorm2d(out_channel),
#             nn.ReLU(),
#         )
#         self.dep = nn.Sequential(
#             nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channel),
#             nn.ReLU(),
#         )
#         self.rgb_branch1 = nn.Conv2d(out_channel, out_channel//2, 3, padding=1, dilation=1)
#         self.rgb_branch2 = nn.Conv2d(out_channel, out_channel//2, 3, padding=3, dilation=3)
#         self.rgb_branch3 = nn.Conv2d(out_channel, out_channel//2, 3, padding=5, dilation=5)
#         self.rgb_branch4 = nn.Conv2d(out_channel, out_channel//2, 3, padding=7, dilation=7)
#
#         self.d_branch1 = nn.Conv2d(out_channel, out_channel//2, 3, padding=1, dilation=1)
#         self.d_branch2 = nn.Conv2d(out_channel, out_channel//2, 3, padding=3, dilation=3)
#         self.d_branch3 = nn.Conv2d(out_channel, out_channel//2, 3, padding=5, dilation=5)
#         self.d_branch4 = nn.Conv2d(out_channel, out_channel//2, 3, padding=7, dilation=7)
#
#         self.CMF1 = CMF(out_channel,ratio)
#         self.CMF2 = CMF(out_channel,ratio)
#         self.CMF3 = CMF(out_channel,ratio)
#         self.CMF4 = CMF(out_channel,ratio)
#
#         self.conv = nn.Sequential(
#             nn.Conv2d(out_channel*4, out_channel,kernel_size=1),
#             nn.Conv2d(out_channel,out_channel,kernel_size=3,padding=1),
#             nn.BatchNorm2d(out_channel),
#             nn.ReLU(),
#         )
#         # self.conv_rgb = nn.Conv2d(out_channel,out_channel,kernel_size=3,padding=1)
#     def forward(self,rgb,dep):
#         rgb = self.rgb(rgb)
#         dep = self.dep(dep)
#
#         x1_rgb = self.rgb_branch1(rgb)
#         x2_rgb = self.rgb_branch2(rgb)
#         x3_rgb = self.rgb_branch3(rgb)
#         x4_rgb = self.rgb_branch4(rgb)
#
#         x1_d = self.d_branch1(dep)
#         x2_d = self.d_branch2(dep)
#         x3_d = self.d_branch3(dep)
#         x4_d = self.d_branch4(dep)
#
#         e111 = self.CMF1(x1_rgb,x1_d)
#         e222 = self.CMF2(x2_rgb,x2_d)
#         e333 = self.CMF3(x3_rgb,x3_d)
#         e444 = self.CMF4(x4_rgb,x4_d)
#
#         e_fuse = torch.cat((e111, e222, e333, e444), dim=1)
#         # e_fuse = self.conv(e_fuse) + self.conv_rgb(rgb)
#         e_fuse = self.conv(e_fuse)
#
#         return e_fuse
#
# class DecoderBlock(nn.Module):
#     def __init__(self,in_channels, n_filters):
#         super(DecoderBlock, self).__init__()
#         # B, C, H, W -> B, C/4, H, W
#         self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
#         self.norm1 = nn.BatchNorm2d(in_channels // 4)
#         self.relu1 = nn.ReLU(inplace=True)
#
#         # B, C/4, H, W -> B, C/4, H, W
#         self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3,
#                                           stride=2, padding=1, output_padding=1)
#         self.norm2 = nn.BatchNorm2d(in_channels // 4)
#         self.relu2 = nn.ReLU(inplace=True)
#
#         # B, C/4, H, W -> B, C, H, W
#         self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
#         self.norm3 = nn.BatchNorm2d(n_filters)
#         self.relu3 = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.norm1(x)
#         x = self.relu1(x)
#         x = self.deconv2(x)
#         x = self.norm2(x)
#         x = self.relu2(x)
#         x = self.conv3(x)
#         x = self.norm3(x)
#         x = self.relu3(x)
#         return x
#
#
# class SFAFMA_S(nn.Module):
#     def __init__(self):
#         super(SFAFMA_S, self).__init__()
#         self.mobilenetv2_rgb = mobilenet_v2()
#         self.mobilenetv2_dep = mobilenet_v2()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False)
#
#         ##########   ENCODER    ###########
#         # self.HLG0 = HLG(16,16,ratio=2)
#         # self.HLG1 = HLG(24,24,ratio=4)
#         # self.HLG2 = HLG(32,32,ratio=4)
#         # self.HLG3 = HLG(160,60,ratio=16)
#         # self.HLG4 = HLG(320,100,ratio=16)
#         self.fuse0 = BasicConv2d(16,16, kernel_size=3, padding=1)
#         self.fuse1 = BasicConv2d(24,24, kernel_size=3, padding=1)
#         self.fuse2 = BasicConv2d(32,32, kernel_size=3, padding=1)
#         self.fuse3 = BasicConv2d(160,60, kernel_size=3, padding=1)
#         self.fuse4 = BasicConv2d(320,100, kernel_size=3, padding=1)
#
#
#
#        ##########   DECODER    ###########
#         self.decoder4 = DecoderBlock(100, 60)
#         self.decoder3 = DecoderBlock(60, 32)
#         self.decoder2 = DecoderBlock(32, 24)
#         self.decoder1 = DecoderBlock(24, 16)
#         self.decoder0 = DecoderBlock(16, 10)
#
#         self.S4 = nn.Conv2d(60, 6, 3, stride=1, padding=1)
#         self.S3 = nn.Conv2d(32, 6, 3, stride=1, padding=1)
#         self.S2 = nn.Conv2d(24, 6, 3, stride=1, padding=1)
#         self.S1 = nn.Conv2d(16, 6, 3, stride=1, padding=1)
#         self.S0 = nn.Conv2d(10, 6, 3, stride=1, padding=1)
#
#     def forward(self, rgb,dep):
#         x0 = self.mobilenetv2_rgb.features[0:2](rgb)
#         x1 = self.mobilenetv2_rgb.features[2:4](x0)
#         x2 = self.mobilenetv2_rgb.features[4:7](x1)
#         x3 = self.mobilenetv2_rgb.features[7:17](x2)
#         x4 = self.mobilenetv2_rgb.features[17:18](x3)
#
#         x_d = self.conv1(dep)
#         d0 = self.mobilenetv2_dep.features[1:2](x_d)
#         d1 = self.mobilenetv2_dep.features[2:4](d0)
#         d2 = self.mobilenetv2_dep.features[4:7](d1)
#         d3 = self.mobilenetv2_dep.features[7:17](d2)
#         d4 = self.mobilenetv2_dep.features[17:18](d3)
#
#
#         # E0 = self.HLG0(x0, d0)
#         # E1 = self.HLG1(x1, d1)
#         # E2 = self.HLG2(x2, d2)
#         # E3 = self.HLG3(x3, d3)
#         # E4 = self.HLG4(x4, d4)
#         E0 = self.fuse0(x0 + d0)
#         E1 = self.fuse1(x1 + d1)
#         E2 = self.fuse2(x2 + d2)
#         E3 = self.fuse3(x3 + d3)
#         E4 = self.fuse4(x4 + d4)
#         # print(E0.shape)
#         # print(E1.shape)
#         # print(E2.shape)
#         # print(E3.shape)
#         # print(E4.shape)
#         ##########   DECODER    ###########
#         d4 = self.decoder4(E4) + E3
#         z4 = self.S4(d4)
#
#         d3 = self.decoder3(d4) + E2
#         z3 = self.S3(d3)
#
#         d2 = self.decoder2(d3) + E1
#         z2 = self.S2(d2)
#
#         d1 = self.decoder1(d2) + E0
#         z1 = self.S1(d1)
#
#         d0 = self.decoder0(d1)
#         z0 = self.S0(d0)
#         # fuse1 = self.finaldeconv1(d1)
#         # fuse2 = self.finalrelu1(fuse1)
#         # t0 = self.finalconv2(fuse2)
#
#         # t0,t1,t2,t3,t4 = self.decoder(E0,E1,E2,E3,E4)
#
#         # Z0 = F.interpolate(t0,size=(256,256),mode='bilinear')
#         # Z1 = F.interpolate(t1,size=(256,256),mode='bilinear')
#         # Z2 = F.interpolate(t2,size=(256,256),mode='bilinear')
#         # Z3 = F.interpolate(t3,size=(256,256),mode='bilinear')
#         # Z4 = F.interpolate(t4,size=(256,256),mode='bilinear')
#         # t5 = F.interpolate(t5,size=(256,256),mode='bilinear')
#
#         return z0,z1,z2,z3,z4
#         # return z0,z1,z2,z3,z4,d0, d1, d2, d3, d4, E0, E1, E2, E3, E4



if __name__ == '__main__':
    rgb = torch.randn(10,3,256,256)
    dep = torch.randn(10,1,256,256)
    net = SFAFMA_S()
    out = net(rgb,dep)
    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)
    print(out[3].shape)
    print(out[4].shape)

    # PPNet_11-v