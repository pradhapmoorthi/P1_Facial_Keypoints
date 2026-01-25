import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)    # (224 -> 220)
        self.pool1 = nn.MaxPool2d(2, 2)                # (220 -> 110)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  # (110 -> 108)
        self.pool2 = nn.MaxPool2d(2, 2)                # (108 -> 54)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3) # (54 -> 52)
        self.pool3 = nn.MaxPool2d(2, 2)                # (52 -> 26)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)# (26 -> 24)
        self.pool4 = nn.MaxPool2d(2, 2)                # (24 -> 12)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 12 * 12, 1000)
        self.fc2 = nn.Linear(1000, 136)  # 68 keypoints * 2
        
        # Dropout for regularization
        self.drop = nn.Dropout(p=0.4)

    def forward(self, x):
        # Convolution + ReLU + Pooling
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with dropout
        x = self.drop(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x
