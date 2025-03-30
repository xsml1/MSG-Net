import torch
import torch.nn as nn
import torch.nn.functional as F

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)
def catcat3(inputs1, inputs2, inputs3):
    return torch.cat((inputs1, inputs2, inputs3), 1)

class Cat3(nn.Module):
    def __init__(self):
        super(Cat3, self).__init__()

    def forward(self, x1, x2, x3):
        return catcat3(x1, x2, x3)

class PixelwiseNorm(nn.Module):
    def __init__(self):
        super(PixelwiseNorm, self).__init__()

    def forward(self, x, alpha=1e-8):
        """
        forward pass of the module
        :param x: input activations volume
        :param alpha: small number for numerical stability
        :return: y => pixel normalized activations
        """
        y = x.pow(2.).mean(dim=1, keepdim=True).add(alpha).sqrt()  # [N1HW]
        y = x / y  # normalize the input x volume
        return y
class conv_block(nn.Module):
    """
    Convolution Block
    with two convolution layers
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv_1 = nn.Conv2d(in_ch, out_ch, (1, 1),
                                padding=0, bias=True)
        self.conv_2 = nn.Conv2d(out_ch, out_ch, (3, 3),
                                padding=1, bias=True)
        self.conv_3 = nn.Conv2d(out_ch, out_ch, (3, 3),
                                padding=1, bias=True)

        # pixel_wise feature normalizer:
        self.pixNorm = PixelwiseNorm()

        # leaky_relu:
        self.relu = nn.GELU()

    def forward(self, x):
        """
        forward pass of the block
        :param x: input
        :return: y => output
        """
        from torch.nn.functional import interpolate

        # y = interpolate(x, scale_factor=2)
        y = self.conv_1(self.relu(self.pixNorm(x)))
        residual = y
        y = self.conv_2(self.relu(self.pixNorm(y)))
        y = self.conv_3(self.relu(self.pixNorm(y)))
        y = y + residual

        return y


def catcat2(inputs1, inputs2):
    return torch.cat((inputs1, inputs2), 1)

class Cat2(nn.Module):
    def __init__(self):
        super(Cat2, self).__init__()

    def forward(self, x1, x2):
        return catcat2(x1, x2)


class se_block(nn.Module):
    def __init__(self, inplanes, reduction=16):
        super(se_block, self).__init__()
        self.conv1x1 = default_conv(inplanes, inplanes, 1, bias=True)
        self.conv3x3 = default_conv(inplanes, inplanes, 3, bias=True)
        self.act1 = nn.GELU()
        self.sig = nn.Sigmoid()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # 自适应平均池化，输出一维向量
            nn.Conv2d(inplanes, inplanes//reduction, 1, 1, 0),# 与下面注释效果一致
            # nn.Linear(inplanes, inplanes // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes // reduction, inplanes, 1, 1, 0),# 与下面注释效果一致
            # nn.Linear(inplanes // reduction, inplanes, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        res = self.conv3x3(x)
        res = self.act1(res)
        res1 = self.conv3x3(res)
        res2 = self.conv1x1(res)
        res2 = self.sig(res2)
        res1 = res1 * res2
        res11 = self.se(res1)
        res11 = res * res11
        res11 = x + res11
        res11 = res1 + res11
        return res11

class Channel_Atten_Fusion(nn.Module):
    """ Layer attention module"""

    def __init__(self, input_dim, bias=True):
        super(Channel_Atten_Fusion, self).__init__()
        self.chanel_in = 3*input_dim
        self.skip = nn.Conv2d(in_channels=self.chanel_in, out_channels=self.chanel_in, kernel_size=1)
        self.temperature = nn.Parameter(torch.ones(1))

        self.qkv = nn.Conv2d(self.chanel_in, self.chanel_in * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(self.chanel_in * 3, self.chanel_in * 3, kernel_size=3, stride=1, padding=1,
                                    groups=self.chanel_in * 3, bias=bias)
        self.project_out = nn.Conv2d(self.chanel_in, self.chanel_in, kernel_size=1, bias=bias)

        self.conv_out = conv_block(in_ch=self.chanel_in, out_ch=input_dim)

    def forward(self, x1, x2, x3):

        x = torch.cat([x1, x2, x3], dim=1)  # (1, 3, 64, 256, 256)
        m_batchsize, NC, height, width = x.size()


        x_skip = self.skip(x)
        qkv = self.qkv_dwconv(self.qkv(x))  # 1, 192*3, 256, 256
        q, k, v = qkv.chunk(3, dim=1)  # 1, 192, 256, 256
        q = q.view(m_batchsize, NC, -1)  # 1, 3*64, 256*256
        k = k.view(m_batchsize, NC, -1)
        v = v.view(m_batchsize, NC, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature  # 1, 3*64, 3*64
        attn = attn.softmax(dim=-1)

        out_1 = (attn @ v)  # 1, 3*64, 256*256
        out_1 = out_1.view(m_batchsize, -1, height, width)  # 1, 3*64, 256, 256

        out_1 = self.project_out(out_1)
        out_1 = out_1.view(m_batchsize, -1, height, width)  # (1, 192, 256, 256)

        out = out_1 + x

        out = self.conv_out(out)

        return out

from Model.CGAT import Transformer

class MVE_net(nn.Module):

    '''
    修改resdule -model模块 == >> channel group axis_based transformer
    并修改了ffn模块
    从原来的卷积特征融合变成了Channel_Atten_Fusion
    '''


    def __init__(self, input_nc=64, ngf=32, use_dropout=False, padding_type='reflect'):
        super(MVE_net, self).__init__()
        self.projecte_in = default_conv(3, 64, 1, bias=True)
        self.projecte_out = default_conv(64, 3, 3, bias=True)

        self.Trans_block1 = Transformer(dim=64, num_heads=2, num_groups=4, num_layers=1)
        self.Trans_block2 = Transformer(dim=128, num_heads=2, num_groups=8, num_layers=2)
        self.Trans_block3 = Transformer(dim=192, num_heads=2, num_groups=12, num_layers=4)

        self.conv_1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
        self.Fusion_block = Channel_Atten_Fusion(input_dim=64)
        self.Se_block = se_block(inplanes=192, reduction=16)
        self.Se_block1 = se_block(inplanes=384, reduction=16)
        self.conv1 = default_conv(192, 64, 1, bias=True)
        self.conv11 = default_conv(384, 192, 1, bias=True)

        self.cat3 = Cat3()
        self.cat2 = Cat2()
        ###### downsample
        self.down1 = nn.Sequential(nn.Conv2d(input_nc, ngf*2, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))
        self.down2 = nn.Sequential(nn.Conv2d(ngf*2, ngf * 4, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))

        ###### upsample
        self.up1 = nn.Sequential(nn.ConvTranspose2d(ngf * 6, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(ngf * 2, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True))

    def forward(self, inputs):
        x1_, x2_, x3_ = torch.chunk(inputs, chunks=3, dim=1)  # 3, 256, 256
        x1_, x2_, x3_ = self.projecte_in(x1_), self.projecte_in(x2_), self.projecte_in(x3_)


        fusion1 = self.Fusion_block(x1_, x2_, x3_)
        fusion2 = fusion1
        fusion3 = fusion1  # 64, 256, 256

        x3 = self.Trans_block1(fusion3)  # [1, 64, 256, 256]


        x2 = self.cat2(fusion2, x3)
        x2 = self.conv_1(x2)
        x2 = self.Trans_block1(x2)  # [1, 64, 256, 256]

        x1 = self.cat2(fusion1, x2)
        x1 = self.conv_1(x1)
        x1 = self.Trans_block1(x1)  # [1, 64, 256, 256]


        inputs1 = x1_
        inputs2 = x2_
        inputs3 = x3_

        x1 = x1 + inputs1
        x2 = x2 + inputs2
        x3 = x3 + inputs3

        x11 = self.down1(x1)
        x111 = self.down2(x11)
        x22 = self.down1(x2)
        x222 = self.down2(x22)
        x33 = self.down1(x3)
        x333 = self.down2(x33)

        cat1 = self.cat3(x1, x2, x3)  # 64*3
        cat2 = self.cat3(x11, x22, x33)  # 64*3
        cat3 = self.cat3(x111, x222, x333)  # 128*3

        '''
        cat1.shape after cat block:torch.Size([1, 192, 256, 256])
        cat2.shape after cat block:torch.Size([1, 192, 128, 128])
        cat3.shape after cat block:torch.Size([1, 384, 64, 64])
        '''

        x1 = self.Se_block(cat1)  # [1, 192, 256, 256]

        x1 = self.conv1(x1)  # [1, 64, 256, 256]

        x2 = self.Se_block(cat2)
        x2 = self.conv1(x2)
        x3 = self.Se_block1(cat3)
        x3 = self.conv11(x3)  # [1, 192, 64, 64]
        x3 = self.Trans_block3(x3)

        x3 = self.up1(x3)
        x3 = F.interpolate(x3, x2.size()[2:], mode='bilinear', align_corners=True)

        x2 = self.cat2(x2, x3)  # [1, 128, 128, 128]

        x2 = self.Trans_block2(x2)

        x2 = self.conv_1(x2)
        x2 = self.up2(x2)
        x2 = F.interpolate(x2, x1.size()[2:], mode='bilinear', align_corners=True)
        x1 = self.cat2(x1, x2)  # [1, 128, 256, 256]
        x1 = self.conv_1(x1)
        x1 = self.Trans_block1(x1)

        x1 = self.projecte_out(x1)

        return x1