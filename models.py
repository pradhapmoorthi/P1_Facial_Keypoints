import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    """
    VGG-style model for Udacity Facial Keypoints:
    - Input:  (N, 1, 96, 96)
    - Output: (N, 136)  # 68 keypoints * 2
    """
    def __init__(self, num_keypoints=68, p_drop=0.25):
        super().__init__()
        out_dim = num_keypoints * 2

        # Conv blocks: (Conv3x3 + BN + ReLU) x 2 per block
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),               # 96 -> 48
            nn.Dropout(p_drop)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),               # 48 -> 24
            nn.Dropout(p_drop)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),               # 24 -> 12
            nn.Dropout(p_drop)
        )

        # Optional Block4: keeps capacity without exploding params
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),               # 12 -> 6
            nn.Dropout(p_drop)
        )

        # Head: global average pooling to reduce params
        self.gap = nn.AdaptiveAvgPool2d(1)  # (N, C, 1, 1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, out_dim)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)     # keep for 96x96; remove if you use larger inputs
        x = self.gap(x)
        x = self.fc(x)
        return x
