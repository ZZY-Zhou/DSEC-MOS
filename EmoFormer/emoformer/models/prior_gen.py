# torch libraries
import torch
import torch.nn as nn

from .prior_fus import Prompt_block


class ConvBR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super(ConvBR, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False, groups=groups)

        self.bn = nn.BatchNorm2d(out_channel)
        
        self.relu = nn.ReLU()
        
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class PriorGen(nn.Module):
    def __init__(self, in_channel):
        super(PriorGen, self).__init__()
        
        expand_ratio=4.
        
        hidden_channel = int(in_channel * expand_ratio)
        self.conv1 = ConvBR(in_channel, hidden_channel, kernel_size=1, stride=1, padding=0)    ## 1x1 convolution, pw
        self.conv2 = ConvBR(hidden_channel, hidden_channel, kernel_size=3, stride=1, padding=1, groups=hidden_channel) ## depth-wise convolution
        self.conv3 = ConvBR(hidden_channel, in_channel, kernel_size=1, stride=1, padding=0)    ## 1x1 convolution, point-wise convolution, linear

        self.conv_out = ConvBR(in_channel, 1, kernel_size=1, stride=1, padding=0)

        ## Fusion RGB Prior
        self.prior_fus = Prompt_block(inplanes=in_channel, hide_channel=8, smooth=True)

    def forward(self, x):
    
        feat = self.conv1(x)
        feat = self.conv2(feat)
        xg = self.conv3(feat)
        
        pg = self.conv_out(xg)

        x = self.prior_fus(x, xg)

        return x, pg
