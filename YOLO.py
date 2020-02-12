import torch
from  ModelFactory import get_arch
from torch import nn
import math
import torch.nn.functional as F
from torch.nn import init 
import numpy as np

class YOLO(torch.nn.Module):

    def __init__(self, arch_model= 'resnet50', s =14, classes = 20, b = 2, training = True):
        super().__init__()
        self.training = training
        self.classes = classes
        self.s = s
        self.b = b
        _, channels = get_arch(arch_model)
        self.out = nn.Conv2d(channels, self.b*5 + self.classes,3,stride=1, padding=1, bias=False)
        
        init.xavier_normal_(self.out.weight)
        
        self.bn_end = nn.BatchNorm2d( self.b*5 + self.classes)

        for m in self.modules():
            print(m)
            #if isinstance(m, nn.Conv2d):
            #    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            #    m.weight.data.normal_(0, math.sqrt(2. / n))
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.arch,_ = get_arch(arch_model)
    def forward(self, X ):
        feat = self.arch(X)
        out = self.out(feat)
        out_bn = torch.sigmoid(self.bn_end(out))
        return out_bn.permute(0,2,3,1)

