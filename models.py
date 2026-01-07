
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, n_out=136, in_ch=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2)

        # Replace 1x1 with a 3x3 for larger RF (optional)
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        self.pool4 = nn.MaxPool2d(2)

        # GAP-based head (robust to input size)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, n_out)
            # nn.Tanh()  # optionally, if you train targets in [-1, 1]
        )

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.pool4(self.conv4(x))
        x = self.gap(x)
        x = self.head(x)
        return x
