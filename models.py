import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I
from torchvision import models
from collections import OrderedDict

class resnet18(nn.Module):
    def __init__(self):
        super(resnet18, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=(9, 9), stride=(2, 2), padding=(3, 3), bias=False)
        n_inputs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(n_inputs, 136)
                        
    def forward(self, x):
        x = self.resnet18(x)
        return x
