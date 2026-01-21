import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # 1. Convolutional layers
        # Input image size: 224x224 (after transformations)
        # Input channels: 1 (grayscale)

        # Layer 1: Conv -> BN -> ReLU -> MaxPool
        # Input: 1 channel, 224x224
        # Output: 32 channels, 112x112 (after 2x2 pooling)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.bn1   = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Layer 2: Conv -> BN -> ReLU -> MaxPool
        # Input: 32 channels, 112x112
        # Output: 64 channels, 56x56
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Layer 3: Conv -> BN -> ReLU -> MaxPool
        # Input: 64 channels, 56x56
        # Output: 128 channels, 28x28
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Layer 4: Conv -> BN -> ReLU -> MaxPool
        # Input: 128 channels, 28x28
        # Output: 256 channels, 14x14
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)

        # NOTE: We remove Layer 5 pooling to keep more spatial detail (no pool5).
        # If you still want a 5th conv for capacity (optional), you could add it with stride=1.

        # 2. Fully-connected layers
        # After pool4: feature map is (N, 256, 14, 14) -> flatten to 256*14*14 = 50176
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.dropout1 = nn.Dropout(0.30)

        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.20)

        # Output layer: 68 keypoints * 2 (x, y) = 136
        self.fc3 = nn.Linear(256, 136)

        # (Optional) Good practice: initialize linear layers
        nn.init.xavier_uniform_(self.fc1.weight); nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight); nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        ## Define the feedforward behavior of this model
        ## x is the input image and; your output should be a Batch_size x 68*2 keypoint data (the facial keypoints)

        # Block 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))  # (N, 32, 112, 112)

        # Block 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))  # (N, 64, 56, 56)

        # Block 3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))  # (N, 128, 28, 28)

        # Block 4
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))  # (N, 256, 14, 14)

        # Flatten the output for the fully-connected layers
        x = x.view(x.size(0), -1)                        # (N, 256*14*14 = 50176)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)                                  # (N, 136)

        return x
