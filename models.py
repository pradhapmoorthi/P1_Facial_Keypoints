import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, p_drop=0.3):
        super(Net, self).__init__()
        
        # Convolutional layers (same padding to keep dims stable between pools)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)   # (224 -> 224)
        self.pool1 = nn.MaxPool2d(2, 2)                           # (224 -> 112)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # (112 -> 112)
        self.pool2 = nn.MaxPool2d(2, 2)                           # (112 -> 56)
        
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)  # (56 -> 56)
        self.pool3 = nn.MaxPool2d(2, 2)                           # (56 -> 28)
        
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)  # (28 -> 28)
        self.pool4 = nn.MaxPool2d(2, 2)                           # (28 -> 14)
        
        # Flatten dimension after pool4: 32 * 14 * 14 = 6272
        self.fc1  = nn.Linear(32 * 14 * 14, 256)
        self.fc2  = nn.Linear(256, 136)  # 68 keypoints * 2
        
        self.drop = nn.Dropout(p=p_drop)

        # Optional: good initializations (keeps your style intact)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Convolution + ReLU + Pooling
        x = self.pool1(F.relu(self.conv1(x)))  # (N, 16, 112, 112)
        x = self.pool2(F.relu(self.conv2(x)))  # (N, 32, 56, 56)
        x = self.pool3(F.relu(self.conv3(x)))  # (N, 32, 28, 28)
        x = self.pool4(F.relu(self.conv4(x)))  # (N, 32, 14, 14)
        
        # Flatten
        x = x.view(x.size(0), -1)              # (N, 6272)
        
        # Fully connected layers with dropout
        x = self.drop(F.relu(self.fc1(x)))     # (N, 256)
        x = self.fc2(x)                        # (N, 136)
        return x
``
