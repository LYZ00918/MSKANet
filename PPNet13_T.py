import torch
import torch.nn as nn
import torch.nn.functional as F
from toolbox.backbone.ResNet import Backbone_ResNet50_in3,Backbone_ResNet50_in1
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
        e_fuse = self.conv(e_fuse)

        return e_fuse

class Mul_Agg_3(nn.Module):
    def __init__(self, channel1, channel2, channel3, index=1):
        super(Mul_Agg_3, self).__init__()
        self.U = nn.Sequential(
            nn.Conv2d(channel3, channel2, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )
        self.D_M = nn.Sequential(
            nn.Conv2d(channel1, channel2, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.D_A = nn.Sequential(
            nn.Conv2d(channel1, channel2, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(channel2 * 3, channel2, kernel_size=1),
            nn.BatchNorm2d(channel2),
            nn.ReLU(),
        )
        self.index = index
        # 最大池化 用于浅层 ，平均池化 用于高层

    def forward(self, x1, x2, x3):
        if self.index == 1:
            x1 = self.D_A(x1)
        else:
            x1 = self.D_M(x1)
        x3 = self.U(x3)
        fuse = torch.cat((x1, x2, x3), dim=1)
        fuse = self.conv(fuse)
        return fuse


class Mul_Agg_2(nn.Module):
    def __init__(self, channel1, channel2, index=1):
        super(Mul_Agg_2, self).__init__()
        self.U = nn.Sequential(
            nn.Conv2d(channel2, channel1, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )
        self.D_A = nn.Sequential(
            nn.Conv2d(channel1, channel2, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(channel1 * 2, channel1, kernel_size=1),
            nn.BatchNorm2d(channel1),
            nn.ReLU(),
        )
        self.index = index
        # 最大池化 用于浅层 ，平均池化 用于高层

    def forward(self, x1, x2):
        if self.index == 1:
            x2 = self.U(x2)
        else:
            x1 = self.D_A(x1)

        fuse = torch.cat((x1, x2), dim=1)
        fuse = self.conv(fuse)
        return fuse
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

class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        self.agg0 = Mul_Agg_2(64, 128, index=1)
        self.agg1 = Mul_Agg_3(64, 128, 256, index=2)
        self.agg2 = Mul_Agg_3(128, 256, 512, index=2)
        self.agg3 = Mul_Agg_3(256, 512, 512, index=1)
        self.agg4 = Mul_Agg_2(512, 512, index=2)
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )
        self.conv4 = nn.Conv2d(1024, 512, kernel_size=1)

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )
        self.conv3 = nn.Conv2d(768, 256, kernel_size=1)

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )
        self.conv2 = nn.Conv2d(384, 128, kernel_size=1)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )
        self.conv1 = nn.Conv2d(192, 64, kernel_size=1)
        self.conv0 = nn.Conv2d(64, 32, kernel_size=1)

    def forward(self, E0, E1, E2, E3, E4):
        x0 = self.agg0(E0, E1)
        x1 = self.agg1(E0, E1, E2)
        x2 = self.agg2(E1, E2, E3)
        x3 = self.agg3(E2, E3, E4)
        x4 = self.agg4(E3, E4)

        z4 = self.conv4(torch.cat((x3, self.up4(x4)), dim=1))
        z3 = self.conv3(torch.cat((x2, self.up3(z4)), dim=1))
        z2 = self.conv2(torch.cat((x1, self.up2(z3)), dim=1))
        z1 = self.conv1(torch.cat((x0, self.up2(z2)), dim=1))
        z0 = self.conv0(self.up2(z1))
        return z0,z1,z2,z3,z4
class SFAFMA_T(nn.Module):
    def __init__(self):
        super(SFAFMA_T, self).__init__()
        (
            self.encoder1,
            self.encoder2,
            self.encoder4,
            self.encoder8,
            self.encoder16,
        ) = Backbone_ResNet50_in3()
        (
            self.depth_encoder2,
            self.depth_encoder4,
            self.depth_encoder8,
            self.depth_encoder16,
            self.depth_encoder32
        ) = Backbone_ResNet50_in1()

        self.encoder_dep_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.lateral_conv0 = BasicConv2d(64, 64, 3, stride=1, padding=1)
        self.lateral_conv1 = BasicConv2d(256, 128, 3, stride=1, padding=1)
        self.lateral_conv2 = BasicConv2d(512, 256, 3, stride=1, padding=1)

        ##########   ENCODER    ###########
        self.HLG0 = HLG(64,64,ratio=4)
        self.HLG1 = HLG(256,128,ratio=16)
        self.HLG2 = HLG(512,256,ratio=16)
        self.HLG3 = HLG(1024,512,ratio=16)
        self.HLG4 = HLG(2048,512,ratio=16)

       ##########   DECODER    ###########
        self.decoder4 = DecoderBlock(512, 512)
        self.decoder3 = DecoderBlock(512, 256)
        self.decoder2 = DecoderBlock(256, 128)
        self.decoder1 = DecoderBlock(128, 64)
        self.decoder0 = DecoderBlock(64, 32)
        self.finaldeconv1 = nn.ConvTranspose2d(64, 32, 3, stride=2)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, 6, 2, padding=1)


        self.decoder = decoder()
        self.S4 = nn.Conv2d(512, 6, 3, stride=1, padding=1)
        self.S3 = nn.Conv2d(256, 6, 3, stride=1, padding=1)
        self.S2 = nn.Conv2d(128, 6, 3, stride=1, padding=1)
        self.S1 = nn.Conv2d(64, 6, 3, stride=1, padding=1)
        self.S0 = nn.Conv2d(32, 6, 3, stride=1, padding=1)

    def forward(self, rgb,dep):
        x0 = self.encoder1(rgb)
        x1 = self.encoder2(x0)
        x2 = self.encoder4(x1)
        x3 = self.encoder8(x2)
        x4 = self.encoder16(x3)

        # dep = self.encoder_dep_conv1(dep)
        d0 = self.depth_encoder2(dep)
        d1 = self.depth_encoder4(d0)
        d2 = self.depth_encoder8(d1)
        d3 = self.depth_encoder16(d2)
        d4 = self.depth_encoder32(d3)


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

        # Z0 = F.interpolate(t0,size=(256,256),mode='bilinear')
        # Z1 = F.interpolate(t1,size=(256,256),mode='bilinear')
        # Z2 = F.interpolate(t2,size=(256,256),mode='bilinear')
        # Z3 = F.interpolate(t3,size=(256,256),mode='bilinear')
        # Z4 = F.interpolate(t4,size=(256,256),mode='bilinear')
        # t5 = F.interpolate(t5,size=(256,256),mode='bilinear')

        return z0,z1,z2,z3,z4
        # return z0,z1,z2,z3,z4,d0,d1,d2,d3,d4,E0,E1,E2,E3,E4

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
#         e_fuse = self.conv(e_fuse)
#
#         return e_fuse
#
# class Mul_Agg_3(nn.Module):
#     def __init__(self, channel1, channel2, channel3, index=1):
#         super(Mul_Agg_3, self).__init__()
#         self.U = nn.Sequential(
#             nn.Conv2d(channel3, channel2, kernel_size=3, padding=1),
#             nn.BatchNorm2d(channel2),
#             nn.ReLU(inplace=True),
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#         )
#         self.D_M = nn.Sequential(
#             nn.Conv2d(channel1, channel2, kernel_size=3, padding=1),
#             nn.BatchNorm2d(channel2),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#         self.D_A = nn.Sequential(
#             nn.Conv2d(channel1, channel2, kernel_size=3, padding=1),
#             nn.BatchNorm2d(channel2),
#             nn.ReLU(),
#             nn.AvgPool2d(kernel_size=2, stride=2)
#         )
#         self.conv = nn.Sequential(
#             nn.Conv2d(channel2 * 3, channel2, kernel_size=1),
#             nn.BatchNorm2d(channel2),
#             nn.ReLU(),
#         )
#         self.index = index
#         # 最大池化 用于浅层 ，平均池化 用于高层
#
#     def forward(self, x1, x2, x3):
#         if self.index == 1:
#             x1 = self.D_A(x1)
#         else:
#             x1 = self.D_M(x1)
#         x3 = self.U(x3)
#         fuse = torch.cat((x1, x2, x3), dim=1)
#         fuse = self.conv(fuse)
#         return fuse
#
#
# class Mul_Agg_2(nn.Module):
#     def __init__(self, channel1, channel2, index=1):
#         super(Mul_Agg_2, self).__init__()
#         self.U = nn.Sequential(
#             nn.Conv2d(channel2, channel1, kernel_size=3, padding=1),
#             nn.BatchNorm2d(channel1),
#             nn.ReLU(inplace=True),
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#         )
#         self.D_A = nn.Sequential(
#             nn.Conv2d(channel1, channel2, kernel_size=3, padding=1),
#             nn.BatchNorm2d(channel2),
#             nn.ReLU(),
#             nn.AvgPool2d(kernel_size=2, stride=2)
#         )
#         self.conv = nn.Sequential(
#             nn.Conv2d(channel1 * 2, channel1, kernel_size=1),
#             nn.BatchNorm2d(channel1),
#             nn.ReLU(),
#         )
#         self.index = index
#         # 最大池化 用于浅层 ，平均池化 用于高层
#
#     def forward(self, x1, x2):
#         if self.index == 1:
#             x2 = self.U(x2)
#         else:
#             x1 = self.D_A(x1)
#
#         fuse = torch.cat((x1, x2), dim=1)
#         fuse = self.conv(fuse)
#         return fuse
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
# class decoder(nn.Module):
#     def __init__(self):
#         super(decoder, self).__init__()
#         self.agg0 = Mul_Agg_2(64, 128, index=1)
#         self.agg1 = Mul_Agg_3(64, 128, 256, index=2)
#         self.agg2 = Mul_Agg_3(128, 256, 512, index=2)
#         self.agg3 = Mul_Agg_3(256, 512, 512, index=1)
#         self.agg4 = Mul_Agg_2(512, 512, index=2)
#         self.up4 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#         )
#         self.conv4 = nn.Conv2d(1024, 512, kernel_size=1)
#
#         self.up3 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#         )
#         self.conv3 = nn.Conv2d(768, 256, kernel_size=1)
#
#         self.up2 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#         )
#         self.conv2 = nn.Conv2d(384, 128, kernel_size=1)
#
#         self.up1 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#         )
#         self.conv1 = nn.Conv2d(192, 64, kernel_size=1)
#         self.conv0 = nn.Conv2d(64, 32, kernel_size=1)
#
#     def forward(self, E0, E1, E2, E3, E4):
#         x0 = self.agg0(E0, E1)
#         x1 = self.agg1(E0, E1, E2)
#         x2 = self.agg2(E1, E2, E3)
#         x3 = self.agg3(E2, E3, E4)
#         x4 = self.agg4(E3, E4)
#
#         z4 = self.conv4(torch.cat((x3, self.up4(x4)), dim=1))
#         z3 = self.conv3(torch.cat((x2, self.up3(z4)), dim=1))
#         z2 = self.conv2(torch.cat((x1, self.up2(z3)), dim=1))
#         z1 = self.conv1(torch.cat((x0, self.up2(z2)), dim=1))
#         z0 = self.conv0(self.up2(z1))
#         return z0,z1,z2,z3,z4
# class SFAFMA_T(nn.Module):
#     def __init__(self):
#         super(SFAFMA_T, self).__init__()
#         (
#             self.encoder1,
#             self.encoder2,
#             self.encoder4,
#             self.encoder8,
#             self.encoder16,
#         ) = Backbone_ResNet50_in3()
#         (
#             self.depth_encoder2,
#             self.depth_encoder4,
#             self.depth_encoder8,
#             self.depth_encoder16,
#             self.depth_encoder32
#         ) = Backbone_ResNet50_in3()
#
#         self.encoder_dep_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
#
#         self.lateral_conv0 = BasicConv2d(64, 64, 3, stride=1, padding=1)
#         self.lateral_conv1 = BasicConv2d(256, 128, 3, stride=1, padding=1)
#         self.lateral_conv2 = BasicConv2d(512, 256, 3, stride=1, padding=1)
#
#         ##########   ENCODER    ###########
#         # self.HLG0 = HLG(64,64,ratio=4)
#         # self.HLG1 = HLG(256,128,ratio=16)
#         # self.HLG2 = HLG(512,256,ratio=16)
#         # self.HLG3 = HLG(1024,512,ratio=16)
#         # self.HLG4 = HLG(2048,512,ratio=16)
#
#         self.fuse0 = BasicConv2d(64,64,kernel_size=3,padding=1)
#         self.fuse1 = BasicConv2d(256,128,kernel_size=3,padding=1)
#         self.fuse2 = BasicConv2d(512,256,kernel_size=3,padding=1)
#         self.fuse3 = BasicConv2d(1024,512,kernel_size=3,padding=1)
#         self.fuse4 = BasicConv2d(2048,512,kernel_size=3,padding=1)
#
#        ##########   DECODER    ###########
#         self.decoder4 = DecoderBlock(512, 512)
#         self.decoder3 = DecoderBlock(512, 256)
#         self.decoder2 = DecoderBlock(256, 128)
#         self.decoder1 = DecoderBlock(128, 64)
#         self.decoder0 = DecoderBlock(64, 32)
#         self.finaldeconv1 = nn.ConvTranspose2d(64, 32, 3, stride=2)
#         self.finalrelu1 = nn.ReLU(inplace=True)
#         self.finalconv2 = nn.Conv2d(32, 32, 3)
#         self.finalrelu2 = nn.ReLU(inplace=True)
#         self.finalconv3 = nn.Conv2d(32, 6, 2, padding=1)
#
#
#         self.decoder = decoder()
#         self.S4 = nn.Conv2d(512, 6, 3, stride=1, padding=1)
#         self.S3 = nn.Conv2d(256, 6, 3, stride=1, padding=1)
#         self.S2 = nn.Conv2d(128, 6, 3, stride=1, padding=1)
#         self.S1 = nn.Conv2d(64, 6, 3, stride=1, padding=1)
#         self.S0 = nn.Conv2d(32, 6, 3, stride=1, padding=1)
#
#     def forward(self, rgb,dep):
#         x0 = self.encoder1(rgb)
#         x1 = self.encoder2(x0)
#         x2 = self.encoder4(x1)
#         x3 = self.encoder8(x2)
#         x4 = self.encoder16(x3)
#
#         # dep = self.encoder_dep_conv1(dep)
#         d0 = self.depth_encoder2(dep)
#         d1 = self.depth_encoder4(d0)
#         d2 = self.depth_encoder8(d1)
#         d3 = self.depth_encoder16(d2)
#         d4 = self.depth_encoder32(d3)
#
#         E0 = self.fuse0(x0+d0)
#         E1 = self.fuse1(x1+d1)
#         E2 = self.fuse2(x2+d2)
#         E3 = self.fuse3(x3+d3)
#         E4 = self.fuse4(x4+d4)
#
#         # E0 = self.HLG0(x0, d0)
#         # E1 = self.HLG1(x1, d1)
#         # E2 = self.HLG2(x2, d2)
#         # E3 = self.HLG3(x3, d3)
#         # E4 = self.HLG4(x4, d4)
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
#
#         # Z0 = F.interpolate(t0,size=(256,256),mode='bilinear')
#         # Z1 = F.interpolate(t1,size=(256,256),mode='bilinear')
#         # Z2 = F.interpolate(t2,size=(256,256),mode='bilinear')
#         # Z3 = F.interpolate(t3,size=(256,256),mode='bilinear')
#         # Z4 = F.interpolate(t4,size=(256,256),mode='bilinear')
#         # t5 = F.interpolate(t5,size=(256,256),mode='bilinear')
#
#         return z0,z1,z2,z3,z4
#         # return z0,z1,z2,z3,z4,d0,d1,d2,d3,d4,E0,E1,E2,E3,E4
if __name__ == '__main__':
    rgb = torch.randn(10,3,256,256)
    dep = torch.randn(10,1,256,256)
    net = SFAFMA_T()
    out = net(rgb,dep)
    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)
    print(out[3].shape)
    print(out[4].shape)

    # PPNet_11-v
    # ppNet_12-v decoder0 change
