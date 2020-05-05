import functools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import os
try:
    # from DCNv2.dcn_v2 import *
    from dcn_v2 import *
except ImportError:
    raise ImportError('Failed to import DCNv2 module.')
class IRDNN_SNR(nn.Module):
    def __init__(self, nf=32, nframes=3):
        super(IRDNN_SNR, self).__init__()

        self.MR_0 = MotionRestorationModule()
        self.MR_1 = MotionRestorationModule()
        self.CR_0 = CompressRestorationModule()
        self.CR_1 = CompressRestorationModule()
        self.fea_fuse  = nn.Conv2d(nf*4, nf*2, 1, 1, 0, bias = True)
        self.FM1 = FusionModule()
        self.FM2 = FusionModule()
        self.rec = nn.Conv2d(nf*2, 1, 3, 1, 1, bias = True)   
        self.relu = nn.ReLU()

    def forward(self, x_center, x_high, x_low):
        '''
        input: 
        [B, N, C, H, W]
        '''

        #Motion Restoration
        fea_MR_0 = self.MR_0(x_center, x_high[:, 0, :, :, :], x_low[:, 0, :, :, :])
        fea_MR_1 = self.MR_1(x_center, x_high[:, 1, :, :, :], x_low[:, 1, :, :, :])

        #Compression Restoration
        fea_CR_0 = self.CR_0(x_center, x_high[:, 0, :, :, :], x_low[:, 0, :, :, :])
        fea_CR_1 = self.CR_1(x_center, x_high[:, 1, :, :, :], x_low[:, 1, :, :, :])

        # feature fusion 
        FM = self.FM2(self.FM1(self.fea_fuse(torch.cat([fea_MR_0, fea_MR_1, fea_CR_0, fea_CR_1], 1))))
        res = self.rec(FM)

        output = x_center + res

        return output

class FEM(nn.Module):

    def __init__(self):
        super(FEM, self).__init__()
        self.ConvBlock1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.ConvBlock2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.ConvBlock3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1)
        )

    def forward(self, input):
        fea_l1 = self.ConvBlock1(input)
        fea_l2 = self.ConvBlock2(fea_l1)
        fea_l3 = self.ConvBlock3(fea_l2)

        return [fea_l1, fea_l2, fea_l3]

class MFEO(nn.Module):
    def __init__(self):
        super(MFEO,self).__init__()
        self.conv_in = nn.Conv2d(2, 32, 3, 1, 1, bias=True)
        self.dilation_1 = nn.Conv2d(32, 32, 3, stride=1, padding=1, dilation=1, bias=True)
        self.dilation_2 = nn.Conv2d(32, 32, 3, stride=1, padding=2, dilation=2, bias=True)
        self.dilation_3 = nn.Conv2d(32, 32, 3, stride=1, padding=4, dilation=4, bias=True)
        self.conv_out = nn.Conv2d(32*3, 32, 1, 1, 0, bias=True)
        self.relu = nn.ReLU()

    def forward(self, f_i, f_i_n):
        #concat offset
        offset = self.relu(self.conv_in(torch.cat([f_i, f_i_n], 1)))
        offset_1 = self.dilation_1(offset)
        offset_2 = self.dilation_2(offset)
        offset_3 = self.dilation_3(offset)
        offset = self.conv_out(torch.cat([offset_1, offset_2, offset_3], 1))

        return offset


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention,self).__init__()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, fea_e_i_n, fea_b_i_n, fea_b_i, offset):
        weight_map = self.sigmoid(torch.mean(offset, dim=1, keepdim=True))
        fea = fea_e_i_n + (fea_b_i_n - fea_b_i) * weight_map

        return fea

class MotionRestorationModule(nn.Module):

    def __init__(self, nf = 32):
        super(MotionRestorationModule, self).__init__()
        self.MFEO = MFEO()
        self.FEM = FEM()
        self.fuse  = nn.Conv2d(nf*3, nf, 1, 1, 0, bias = True)
        self.SA = SpatialAttention()

    def forward(self, b_i, b_i_n, e_i_n):
        offset = self.MFEO(e_i_n, b_i_n)
        fea_b_i = self.FEM(b_i)
        fea_b_i_n = self.FEM(b_i_n)
        fea_e_i_n = self.FEM(e_i_n)

        fea_l1 = self.SA(fea_e_i_n[0], fea_b_i_n[0], fea_b_i[0], offset)
        fea_l2 = self.SA(fea_e_i_n[1], fea_b_i_n[1], fea_b_i[1], offset)
        fea_l3 = self.SA(fea_e_i_n[2], fea_b_i_n[2], fea_b_i[2], offset)
        
        fea_MR = self.fuse(torch.cat([fea_l1, fea_l2, fea_l3], 1))

        return fea_MR


class CompressRestorationModule(nn.Module):
    def __init__(self):
        super(CompressRestorationModule,self).__init__()
        self.MFEO = MFEO()
        self.FEM = FEM()
        self.dcnpack1 = DCN_sep(32, 32, 3, stride=1, padding=1, dilation=1,
                                  deformable_groups= 4)
        self.dcnpack2 = DCN_sep(32, 32, 3, stride=1, padding=1, dilation=1,
                                  deformable_groups= 4)
        self.dcnpack3 = DCN_sep(32, 32, 3, stride=1, padding=1, dilation=1,
                                  deformable_groups= 4)
        self.fuse = nn.Conv2d(32*3, 32, 1, 1, 0, bias = True)

    def forward(self, b_i, b_i_n, e_i_n):
        offset = self.MFEO(b_i, b_i_n)
        fea_b_i = self.FEM(b_i)
        fea_b_i_n = self.FEM(b_i_n)
        fea_e_i_n = self.FEM(e_i_n)
        fea_l1 = fea_b_i[0] + self.dcnpack1(fea_e_i_n[0] - fea_b_i_n[0], offset)
        fea_l2 = fea_b_i[1] + self.dcnpack2(fea_e_i_n[1] - fea_b_i_n[1], offset)
        fea_l3 = fea_b_i[2] + self.dcnpack3(fea_e_i_n[2] - fea_b_i_n[2], offset)

        fea_CR = self.fuse(torch.cat([fea_l1, fea_l2, fea_l3], 1)) 

        return fea_CR


class FusionModule(nn.Module):
    '''Dense block
    for the second denseblock, t_reduced = True'''

    def __init__(self,c = 64):
        super(FusionModule, self).__init__()
        self.conv2d_1 = nn.Conv2d(c, c, 3, 1, 1, bias=True)
        self.conv2d_2 = nn.Conv2d(c*2, c, 1, 1, 0, bias=True)
        self.conv2d_3 = nn.Conv2d(c, c, 3, 1, 1, bias=True)
        self.conv2d_4 = nn.Conv2d(c*3, c, 1, 1, 0, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        #x: [B, C, T, H, W]
        x1 = self.relu(self.conv2d_1(x))

        x2 = torch.cat((x, x1), 1)  #64+64
        x2 = self.relu(self.conv2d_2(x2))
        x2 = self.relu(self.conv2d_3(x2))

        x3 = torch.cat((x, x1, x2), 1) #64+64+64
        x3 = self.conv2d_4(x3)

        return x3
