import torch
import torch.nn as nn
import torch.nn.functional as F

class NetB_GAP(nn.Module):
    """
    Net B (32,64,128,256 conv blocks) + Global Avg Pool head for 68 facial keypoints.
    Input:  N × 1 × H × W  (e.g., 96×96 or 224×224)
    Output: N × 136  (68 (x,y) pairs), in [-1,1] if use_tanh=True
    """

    def __init__(self, num_keypoints: int = 68, dropout_p: float = 0.25, use_tanh: bool = True,
                 norm_layer: str = "bn"):
        super().__init__()
        out_dim = 2 * num_keypoints
        Norm = {
            "bn": lambda c: nn.BatchNorm2d(c),
            "gn": lambda c: nn.GroupNorm(32 if c >= 32 else 1, c),
            "none": lambda c: nn.Identity(),
        }[norm_layer.lower()]

        def conv_block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1, bias=False),
                Norm(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2, bias=False),
            Norm(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        # Deeper feature extractor
        self.conv2 = conv_block(32, 64)    # /2
        self.conv3 = conv_block(64, 128)   # /4
        self.conv4 = conv_block(128, 256)  # /8

        # Global Average Pool + small MLP head
        self.gap = nn.AdaptiveAvgPool2d(1)  # -> (N,256,1,1)
        self.head = nn.Sequential(
            nn.Flatten(1),                  # -> (N,256)
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(256, out_dim)
        )

        self.use_tanh = use_tanh

    def forward(self, x):
        x = self.stem(x)   # (N,32,H/2,W/2)
        x = self.conv2(x)  # (N,64,H/4,W/4)
        x = self.conv3(x)  # (N,128,H/8,W/8)
        x = self.conv4(x)  # (N,256,H/16,W/16)
        x = self.gap(x)    # (N,256,1,1)
        x = self.head(x)   # (N,136)
        if self.use_tanh:
            x = torch.tanh(x)
        return x
