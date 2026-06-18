import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBranch(nn.Module):
    def __init__(self, in_channels, channels=(32, 64, 128), dropout=0.15):
        super().__init__()
        layers = []
        prev = in_channels
        for ch in channels:
            layers.extend(
                [
                    nn.Conv2d(prev, ch, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(ch),
                    nn.SiLU(),
                    nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(ch),
                    nn.SiLU(),
                    nn.MaxPool2d(2),
                    nn.Dropout2d(dropout),
                ]
            )
            prev = ch
        self.net = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = channels[-1]

    def forward(self, x):
        return self.pool(self.net(x)).flatten(1)


class MelGccFusionNet(nn.Module):
    """Two-stream drone model: mel identity cues + GCC-PHAT inter-mic delay cues."""

    def __init__(self, num_classes=5, embedding_dim=256, dropout=0.25):
        super().__init__()
        self.mel_branch = ConvBranch(4, channels=(32, 64, 128), dropout=0.10)
        self.gcc_branch = ConvBranch(6, channels=(32, 64, 128), dropout=0.10)
        fusion_dim = self.mel_branch.out_dim + self.gcc_branch.out_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.x_head = nn.Linear(embedding_dim, 1)
        self.y_head = nn.Linear(embedding_dim, 1)
        self.z_head = nn.Linear(embedding_dim, 1)
        self.class_head = nn.Linear(embedding_dim, num_classes)

    def forward(self, mel, gcc):
        mel_feat = self.mel_branch(mel)
        gcc_feat = self.gcc_branch(gcc)
        feat = self.fusion(torch.cat([mel_feat, gcc_feat], dim=1))
        xyz = torch.cat([self.x_head(feat), self.y_head(feat), self.z_head(feat)], dim=1)
        return xyz, self.class_head(feat)
