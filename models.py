import torch
import torch.nn as nn
import Net(nn.Module):
    def __init__(self, dropout_p=0.5, in_channels=1, out_dims=136):
        super().__init__()
        
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

        self.features = nn.Sequential(
            conv_block(in_channels, 64),   # 224 -> 112
            conv_block(64, 128),          # 112 -> 56
            conv_block(128, 256),         # 56 -> 28
            conv_block(256, 512),         # 28 -> 14
            conv_block(512, 512),         # 14 -> 7
        )

        self.flatten_dim = 512 * 7 * 7  # 25088

        self.regressor = nn.Sequential(
            nn.Linear(self.flatten_dim, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, out_dims),  # 136 = 68 keypoints * 2
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)                  # (B, 512, 7, 7)
        x = x.view(x.size(0), -1)             # (B, 25088)
        x = self.regressor(x)                 # (B, 136)
        return x

# Example:
# model = VGGKeypoint224(in_channels=1, out_dims=136)
# criterion = nn.MSELoss()  # or nn.SmoothL1Loss(beta=1.0)
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)


