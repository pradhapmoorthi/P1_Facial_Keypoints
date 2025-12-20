
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I

class Net(nn.Module):
    """
    - Input : (N, 1, 96, 96) grayscale faces (normalized)
    - Output: (N, 136) -> 68 (x, y) keypoints
    """

    def __init__(self):
        super().__init__()

        # Block 1: 1 -> 64
        self.conv1_1 = nn.Conv2d(1,   64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64,  64, kernel_size=3, padding=1)
        self.pool1   = nn.MaxPool2d(2, 2)   # 96 -> 48

        # Block 2: 64 -> 128
        self.conv2_1 = nn.Conv2d(64,  128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2   = nn.MaxPool2d(2, 2)   # 48 -> 24

        # Block 3: 128 -> 256
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3   = nn.MaxPool2d(2, 2)   # 24 -> 12

        # Block 4: 256 -> 512
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4   = nn.MaxPool2d(2, 2)   # 12 -> 6

        # Make output explicitly 6x6 even if input varies slightly
        self.adapt   = nn.AdaptiveAvgPool2d((6, 6))

        # 512 * 6 * 6 = 18432
        self.fc1 = nn.Linear(512 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 136)

        self.dropout = nn.Dropout(p=0.5)

        self._init_weights()

    def _init_weights(self):
        # Kaiming for conv (ReLU), Xavier for linear
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                I.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                I.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Block 1
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)

        # Block 2
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)

        # Block 3
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = self.pool3(x)

        # Block 4
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = self.pool4(x)

        # Normalize spatial size and flatten
        x = self.adapt(x)                     # -> (N, 512, 6, 6)
        x = x.view(x.size(0), -1)             # -> (N, 18432)

        # Fully connected head for regression
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)                       # -> (N, 136)

        return x


