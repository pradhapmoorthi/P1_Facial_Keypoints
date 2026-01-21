import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, out_dim=136):
        super().__init__()
        # Stem
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # 3x3 instead of 5x5 (optional)
        self.bn1   = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # 224 -> 112

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # 112 -> 56

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # 56 -> 28

        # Use stride-2 conv instead of pool4 to preserve learned downsampling
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # 28 -> 14
        self.bn4   = nn.BatchNorm2d(256)

        # Replace flatten+big FC with GAP head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # (N,256,1,1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, out_dim)  # 136
        )

        # Optional init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))  # (N,32,112,112)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))  # (N,64,56,56)
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))  # (N,128,28,28)
        x = F.relu(self.bn4(self.conv4(x)))              # (N,256,14,14)
        x = self.gap(x)                                  # (N,256,1,1)
        x = self.head(x)                                 # (N,136)
        return x
