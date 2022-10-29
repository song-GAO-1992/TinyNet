import torch 
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self,inchannel,outchannel,stride=1):
        super(ResBlock,self).__init__()
        # self.CONV_1 = nn.Conv2d(inchannel,outchannel,kernel_size = 3,stride = stride,padding = 1,bias = False)
        # self.B_1 = nn.BatchNorm2d(outchannel)
        # self.ReLU_1 = nn.ReLU(inplace=True) # 经过relu后改变输入向量中的值
        # self.CONV_2 = nn.Conv2d(outchannel,outchannel,kernel_size = 3,stride = 1,padding = 1,bias = False)
        # self.B_2 = nn.BatchNorm2d(outchannel)
        self.left = nn.Sequential(
            nn.Conv2d(inchannel,outchannel,kernel_size = 3,stride = stride,padding = 1,bias = False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True), # 经过relu后改变输入向量中的值
            nn.Conv2d(outchannel,outchannel,kernel_size = 3,stride = 1,padding = 1,bias = False),
            nn.BatchNorm2d(outchannel))
        
        self.shortcut = nn.Sequential()
        
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel,outchannel,kernel_size=1, stride=stride,bias=False),
                nn.BatchNorm2d(outchannel))

    def forward(self,x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *input):
        return super().forward(*input)

class BackBone(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.CONV_1 = nn.Conv2d(3,7,)
        self.MAX_POOL_1 = nn.MaxPool2d()
        self


class Neck(nn.Module):
    def __init__(self):
        super().__init__()


class Head(nn.Module):
    def __init__(self):
        super().__init__()