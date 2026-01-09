import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # 1 input image channel (grayscale), 32 output channels, 4x4 square convolution kernel
        # Output size: (W-F+2P)/S + 1 = (224-4+2*1)/1 + 1 = 223
        # After pool (2x2): floor(223/2) = 111
        self.conv1 = nn.Conv2d(1, 32, 4, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Maxpooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # 32 input channels, 64 output channels, 3x3 kernel
        # Output size after conv: (111-3+2*1)/1 + 1 = 111
        # After pool (2x2): floor(111/2) = 55
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # 64 input channels, 128 output channels, 3x3 kernel
        # Output size after conv: (55-3+2*1)/1 + 1 = 55
        # After pool (2x2): floor(55/2) = 27
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # 128 input channels, 256 output channels, 3x3 kernel
        # Output size after conv: (27-3+2*1)/1 + 1 = 27
        # After pool (2x2): floor(27/2) = 13
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # Flattened size before the first fully-connected layer:
        # The output of the last pooling layer is 256 feature maps of size 13x13.
        self.fc1 = nn.Linear(256 * 13 * 13, 1000) # Corrected in_features to 43264
        self.fc2 = nn.Linear(1000, 136)
        
        # dropout layer
        self.drop = nn.Dropout(p=0.4)

    def forward(self, x):
        # four conv/relu + pool layers
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        # flatten image input
        x = x.view(x.size(0), -1) # Reshapes to (batch_size, 256*13*13)
        
        # two linear layers with dropout
        x = self.drop(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x
