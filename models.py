import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 12 * 12, 1000)
        self.fc2 = nn.Linear(1000, 136)  # 68 keypoints * 2
        
        # Dropout
        self.drop = nn.Dropout(p=0.4)

        # -------------------------------------------
        # Initialize all Conv2d + Linear weights here
        # -------------------------------------------
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Kaiming (He) initialization — the correct method for ReLU networks.
        Keeps activations stable as depth increases.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))

        x = x.view(x.size(0), -1)

        x = self.drop(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
