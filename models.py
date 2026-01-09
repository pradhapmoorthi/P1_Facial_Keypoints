import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # 1 input image channel (grayscale), 32 output channels, 4x4 square convolution kernel
        # Output size: (W-F+2P)/S + 1 = (224-4+0)/1 + 1 = 221
        # After pool: 221/2 = 110.5 -> 110 (floor)
        self.conv1 = nn.Conv2d(1, 32, 4, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Maxpooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # 32 input channels, 64 output channels, 3x3 kernel
        # Output size after pool: (110-3+0)/1 + 1 = 108 -> 108/2 = 54
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # 64 input channels, 128 output channels, 3x3 kernel
        # Output size after pool: (54-3+0)/1 + 1 = 52 -> 52/2 = 26
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # 128 input channels, 256 output channels, 3x3 kernel
        # Output size after pool: (26-3+0)/1 + 1 = 24 -> 24/2 = 12
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # Flattened size calculations:
        # For an input image of 224x224:
        # conv1: (224-4)/1 + 1 = 221. MaxPool: 221/2 = 110 (approx)
        # conv2: (110-3)/1 + 1 = 108. MaxPool: 108/2 = 54
        # conv3: (54-3)/1 + 1 = 52. MaxPool: 52/2 = 26
        # conv4: (26-3)/1 + 1 = 24. MaxPool: 24/2 = 12
        # So the output features before flattening will be 256 * 12 * 12 = 36864
        # Let's re-calculate using the provided values: 256 * 5 * 5 = 6400 (This is incorrect based on current layers)
        # The traceback implies the tensor after convolutions is 43264. 
        # Let's assume the previous `models.py` had some slight differences or the computation was wrong.
        # If the input was 224 and filter 4, padding 1, stride 1: (224 - 4 + 2*1)/1 + 1 = 223.
        # Pool 2x2: 223 / 2 = 111 (floor).
        # conv2: (111 - 3 + 2*1)/1 + 1 = 111. Pool 2x2: 111 / 2 = 55 (floor).
        # conv3: (55 - 3 + 2*1)/1 + 1 = 55. Pool 2x2: 55 / 2 = 27 (floor).
        # conv4: (27 - 3 + 2*1)/1 + 1 = 27. Pool 2x2: 27 / 2 = 13 (floor).
        # So, the final output size before flattening is 256 * 13 * 13 = 43264.
        
        # fully-connected layer
        self.fc1 = nn.Linear(256 * 13 * 13, 1000) # Corrected in_features
        self.fc2 = nn.Linear(1000, 136)
        
        # dropout
        self.drop = nn.Dropout(p=0.4)

    def forward(self, x):
        # four conv/relu + pool layers
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        # flatten image input
        x = x.view(x.size(0), -1) # (B, 256*13*13)
        
        # two linear layers with dropout
        x = self.drop(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x
