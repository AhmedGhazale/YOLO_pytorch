import torchvision
import torch.nn as nn
import torch
from collections import OrderedDict


def get_arch(name):
    if name == 'vgg16':
        return (torchvision.models.vgg16_bn(pretrained=True).features, 512)

    if name == 'mobilenet_v2':
        return (torchvision.models.mobilenet_v2(pretrained=True).features, 1280)

    if name == 'densenet121':
        return (torchvision.models.densenet121(pretrained=True).features, 1024)

    if name == 'densenet161':
        return (torchvision.models.densenet121(pretrained=True).features, 2208)

    if name == 'densenet201':
        return( torchvision.models.densenet121(pretrained=True).features, 1920)


    if name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
        model = torch.nn.Sequential(model.conv1,
                                    model.bn1,
                                    model.relu,
                                    model.maxpool,
                                    model.layer1,
                                    model.layer2,
                                    model.layer3,
                                    model.layer4)

        return (model, 2048)

      
      


