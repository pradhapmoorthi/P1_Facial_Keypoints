import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)    # (224 -> 220)
        self.pool1 = nn.MaxPool2d(2, 2)                 # (220 -> 110)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)   # (110 -> 108)
        self.pool2 = nn.MaxPool2d(2, 2)                 # (108 -> 54)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)  # (54 -> 52)
        self.pool3 = nn.MaxPool2d(2, 2)                 # (52 -> 26)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3) # (26 -> 24)
        self.pool4 = nn.MaxPool2d(2, 2)                 # (24 -> 12)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 12 * 12, 1000)
        self.fc2 = nn.Linear(1000, 136)  # 68 keypoints * 2
        
        # Dropout for regularization
        self.drop = nn.Dropout(p=0.4)

        # Xavier (Glorot) initialization with activation-aware gains
        self._initialize_weights()

    def _initialize_weights(self):
        # Use ReLU gain for layers followed by ReLU; gain=1.0 for final linear output
        relu_gain = nn.init.calculate_gain('relu')  # ~sqrt(2)

        # Convolutional layers (followed by ReLU)
        nn.init.xavier_uniform_(self.conv1.weight, gain=relu_gain)
        if self.conv1.bias is not None:
            nn.init.constant_(self.conv1.bias, 0.0)

        nn.init.xavier_uniform_(self.conv2.weight, gain=relu_gain)
        if self.conv2.bias is not None:
            nn.init.constant_(self.conv2.bias, 0.0)

        nn.init.xavier_uniform_(self.conv3.weight, gain=relu_gain)
        if self.conv3.bias is not None:
            nn.init.constant_(self.conv3.bias, 0.0)

        nn.init.xavier_uniform_(self.conv4.weight, gain=relu_gain)
        if self.conv4.bias is not None:
            nn.init.constant_(self.conv4.bias, 0.0)

        # Fully connected layers
        # fc1 is followed by ReLU -> use relu gain
        nn.init.xavier_uniform_(self.fc1.weight, gain=relu_gain)
        if self.fc1.bias is not None:
            nn.init.constant_(self.fc1.bias, 0.0)

        # fc2 is the final linear output -> gain=1.0
        nn.init.xavier_uniform_(self.fc2.weight, gain=1.0)
        if self.fc2.bias is not None:
            nn.init.constant_(self.fc2.bias, 0.0)

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
