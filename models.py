import torch
import torch.nn as nn
from torchvision import models

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Load pretrained ResNet-18
        self.resnet18 = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )

        # Replace the first conv layer to accept 1-channel grayscale input
        old_conv = self.resnet18.conv1
        self.resnet18.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Initialize grayscale conv weights using pretrained RGB weights
        # Take mean across RGB channels → shape: (64, 1, 7, 7)
        with torch.no_grad():
            w = old_conv.weight.data
            self.resnet18.conv1.weight.data = w.mean(dim=1, keepdim=True)

        # Replace final fully connected layer for 136 keypoints
        n_inputs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(n_inputs, 136)

    def forward(self, x):
        return self.resnet18(x)
