
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Convolutional layers (no padding, as in your original)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)     # (224 -> 220)
        self.pool1 = nn.MaxPool2d(2, 2)                  # (220 -> 110)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)    # (110 -> 108)
        self.pool2 = nn.MaxPool2d(2, 2)                  # (108 -> 54)
        
        # After pool2: feature map is 64 x 54 x 54
        self.fc1 = nn.Linear(64 * 54 * 54, 1000)         # 186,624 -> 1,000
        self.fc2 = nn.Linear(1000, 136)
        
        self.drop = nn.Dropout(p=0.4)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))            # -> 32 x 110 x 110
        x = self.pool2(F.relu(self.conv2(x)))            # -> 64 x 54 x 54
        
        # Flatten
        x = x.view(x.size(0), -1)                        # -> 64*54*54
        
        x = self.drop(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
