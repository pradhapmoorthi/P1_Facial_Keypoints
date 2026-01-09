import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 1 x 96 x 96
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=1, padding=1)   # -> 32 x 95 x 95
        self.bn1   = nn.BatchNorm2d(32)
        self.pool  = nn.MaxPool2d(2, 2)                                     # -> 32 x 47 x 47

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # -> 64 x 47 x 47
        self.bn2   = nn.BatchNorm2d(64)                                     # -> 64 x 23 x 23

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1) # -> 128 x 23 x 23
        self.bn3   = nn.BatchNorm2d(128)                                    # -> 128 x 11 x 11

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)# -> 256 x 11 x 11
        self.bn4   = nn.BatchNorm2d(256)                                    # -> 256 x 5 x 5

        self.drop  = nn.Dropout(p=0.4)

        self.fc1   = nn.Linear(256 * 5 * 5, 1000)
        self.fc2   = nn.Linear(1000, 136)

    def forward(self, x):
        # x: (B, 1, 96, 96)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # (B, 32, 47, 47)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # (B, 64, 23, 23)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # (B, 128, 11, 11)
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # (B, 256, 5, 5)

        x = x.view(x.size(0), -1)                       # (B, 256*5*5)
        x = self.drop(F.relu(self.fc1(x)))
        x = self.fc2(x)                                 # (B, 136)
        return x
