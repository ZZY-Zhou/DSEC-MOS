import torch
import torch.nn as nn
import torch.nn.functional as F

class Prompt_block(nn.Module, ):
    def __init__(self, inplanes=None, hide_channel=None, smooth=False):
        super(Prompt_block, self).__init__()
        self.conv0_0 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv0_1 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        
        self.conv2 = nn.Conv2d(in_channels=hide_channel*2, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=hide_channel, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv_out = nn.Conv2d(in_channels=hide_channel*2, out_channels=inplanes, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim=-1)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def forward(self, x0, x1):
        x0 = self.conv0_0(x0)
        
        x1 = self.conv0_1(x1)
        
        x = torch.cat((x0, x1), dim=1)
        x = self.conv2(x)
        x = self.softmax(x)
        x = torch.mul(x, x1)
        x = self.conv3(x)
        x = torch.cat((x, x0), dim=1)
        x = self.conv_out(x)
        return x
