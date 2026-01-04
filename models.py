
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolutional stack with padding to stabilize shapes
        self.conv_1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)  # 224x224 -> 224x224 before pool
        self.bn1 = nn.BatchNorm2d(32)

        self.conv_2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # keep size pre-pool
        self.bn2 = nn.BatchNorm2d(64)

        self.conv_3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Use your conv_4 meaningfully
        self.conv_4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)               # 224->112->56->28
        self.gap  = nn.AdaptiveAvgPool2d((1, 1))     # 256 x 1 x 1
        self.drop = nn.Dropout(0.25)

        # Small head (far fewer params than 173k->1000->1000)
        self.fc_1 = nn.Linear(256, 128)
        self.fc_2 = nn.Linear(128, 136)

        # Kaiming init (optional but recommended)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                I.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                I.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv_1(x))))  # 224 -> 112
        x = self.pool(F.relu(self.bn2(self.conv_2(x))))  # 112 -> 56
        x = self.pool(F.relu(self.bn3(self.conv_3(x))))  # 56  -> 28
        x = F.relu(self.bn4(self.conv_4(x)))             # 28 stays 28
        x = self.gap(x)                                  # -> 1x1
        x = torch.flatten(x, 1)                          # 256
        x = self.drop(F.relu(self.fc_1(x)))
        x = self.fc_2(x)                                 # 136
        return x
