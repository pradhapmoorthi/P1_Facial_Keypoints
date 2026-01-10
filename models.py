import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # 1. Convolutional layers
        # Input image size: 224x224 (after transformations)
        # Input channels: 1 (grayscale)

        # Layer 1: Conv -> ReLU -> MaxPool
        # Input: 1 channel, 224x224
        # Output: 32 channels, 112x112 (after 2x2 pooling)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Layer 2: Conv -> ReLU -> MaxPool
        # Input: 32 channels, 112x112
        # Output: 64 channels, 56x56
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Layer 3: Conv -> ReLU -> MaxPool
        # Input: 64 channels, 56x56
        # Output: 128 channels, 28x28
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Layer 4: Conv -> ReLU -> MaxPool
        # Input: 128 channels, 28x28
        # Output: 256 channels, 14x14
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Layer 5: Conv -> ReLU -> MaxPool
        # Input: 256 channels, 14x14
        # Output: 512 channels, 7x7
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(2, 2)

        # 2. Fully-connected layers
        # Input: 512 channels * 7 * 7 = 25088
        self.fc1 = nn.Linear(512 * 7 * 7, 1024)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.4)

        # Output layer: 68 keypoints * 2 (x, y) = 136
        self.fc3 = nn.Linear(512, 136)


    def forward(self, x):
        ## Define the feedforward behavior of this model
        ## x is the input image and; your output should be a Batch_size x 68*2 keypoint data (the facial keypoints)

        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.pool5(F.relu(self.conv5(x)))

        # Flatten the output for the fully-connected layers
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x
