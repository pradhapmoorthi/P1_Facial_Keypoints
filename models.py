
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5) # Input 224 -> (224-5+1)=220
        self.pool1 = nn.MaxPool2d(2, 2)  # 220 -> 110. Output: 16x110x110

        self.conv2 = nn.Conv2d(16, 32, 3) # 110 -> (110-3+1)=108
        self.pool2 = nn.MaxPool2d(2, 2)  # 108 -> 54. Output: 32x54x54

        # Fully connected layer
        self.fc1 = nn.Linear(32 * 54 * 54, 68 * 2)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc1(x)
        return x

class MediumNet(nn.Module):
    def __init__(self):
        super(MediumNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5) # Input 224 -> 220
        self.pool1 = nn.MaxPool2d(2, 2)  # 220 -> 110. Output: 32x110x110
        self.dropout1 = nn.Dropout(p=0.1)

        self.conv2 = nn.Conv2d(32, 64, 3) # 110 -> 108
        self.pool2 = nn.MaxPool2d(2, 2)  # 108 -> 54. Output: 64x54x54
        self.dropout2 = nn.Dropout(p=0.2)

        self.conv3 = nn.Conv2d(64, 128, 3) # 54 -> 52
        self.pool3 = nn.MaxPool2d(2, 2)  # 52 -> 26. Output: 128x26x26
        self.dropout3 = nn.Dropout(p=0.3)

        self.fc1 = nn.Linear(128 * 26 * 26, 512)
        self.dropout4 = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(512, 68 * 2)

    def forward(self, x):
        x = self.dropout1(self.pool1(F.relu(self.conv1(x))))
        x = self.dropout2(self.pool2(F.relu(self.conv2(x))))
        x = self.dropout3(self.pool3(F.relu(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout4(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1) # Input 224 -> 224
        self.pool1 = nn.MaxPool2d(2, 2) # 224 -> 112. Output: 32x112x112
        self.drop1 = nn.Dropout(0.1)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # 112 -> 112
        self.pool2 = nn.MaxPool2d(2, 2) # 112 -> 56. Output: 64x56x56
        self.drop2 = nn.Dropout(0.2)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1) # 56 -> 56
        self.pool3 = nn.MaxPool2d(2, 2) # 56 -> 28. Output: 128x28x28
        self.drop3 = nn.Dropout(0.3)

        self.conv4 = nn.Conv2d(128, 256, 3, padding=1) # 28 -> 28
        self.pool4 = nn.MaxPool2d(2, 2) # 28 -> 14. Output: 256x14x14
        self.drop4 = nn.Dropout(0.4)

        self.fc1 = nn.Linear(256 * 14 * 14, 1000)
        self.drop5 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1000, 68 * 2)

    def forward(self, x):
        x = self.drop1(self.pool1(F.relu(self.conv1(x))))
        x = self.drop2(self.pool2(F.relu(self.conv2(x))))
        x = self.drop3(self.pool3(F.relu(self.conv3(x))))
        x = self.drop4(self.pool4(F.relu(self.conv4(x))))

        x = x.view(x.size(0), -1)
        x = self.drop5(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class ComplexNet(nn.Module):
    def __init__(self):
        super(ComplexNet, self).__init__()
        # 1 input image channel (grayscale), 64 output channels/feature maps, 3x3 square convolution kernel
        self.conv1_1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2) # (224-2)/2+1 = 112 -> 112x112

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2) # (112-2)/2+1 = 56 -> 56x56

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2) # (56-2)/2+1 = 28 -> 28x28

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2) # (28-2)/2+1 = 14 -> 14x14

        # For VGG-like architectures, it's common to have a 7x7 or 6x6 feature map before fully connected layers
        # Given input 224x224, after 4 maxpools of stride 2: 224 / (2^4) = 224 / 16 = 14
        # So output of pool4 is 512x14x14
        # Let's adjust to common output size for FC layers like 6x6 (VGG-like)
        self.adapt = nn.AdaptiveAvgPool2d((6, 6)) # This will make it 512x6x6

        # fully connected layers
        # 512*6*6 = 18432
        self.fc1 = nn.Linear(512 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 68*2) # 68 keypoints * 2 coordinates

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)
        x = self.dropout(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)
        x = self.dropout(x)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = self.pool3(x)
        x = self.dropout(x)

        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = self.pool4(x)
        x = self.dropout(x)

        # flatten features
        x = self.adapt(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


