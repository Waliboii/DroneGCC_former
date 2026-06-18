import math

import torch
import torch.nn as nn


class ConvBranch(nn.Module):
    def __init__(self, in_channels, channels=(32, 64, 128), dropout=0.10):
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


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=32):
        super().__init__()
        pos = torch.arange(max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class CoordHead(nn.Module):
    def __init__(self, dim, hidden=None, dropout=0.10):
        super().__init__()
        if hidden is None:
            self.net = nn.Linear(dim, 1)
        else:
            self.net = nn.Sequential(
                nn.Linear(dim, hidden),
                nn.LayerNorm(hidden),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )

    def forward(self, x):
        return self.net(x)


class MelGccWindowEncoder(nn.Module):
    def __init__(self, embedding_dim=256, dropout=0.20):
        super().__init__()
        self.mel_branch = ConvBranch(4, dropout=0.10)
        self.gcc_branch = ConvBranch(6, dropout=0.10)
        self.fusion = nn.Sequential(
            nn.Linear(self.mel_branch.out_dim + self.gcc_branch.out_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
        )

    def forward(self, mel, gcc):
        return self.fusion(torch.cat([self.mel_branch(mel), self.gcc_branch(gcc)], dim=1))


class MelGccTemporalTransformer(nn.Module):
    def __init__(self, num_classes=5, embedding_dim=256, heads=4, layers=2, dropout=0.15):
        super().__init__()
        self.window_encoder = MelGccWindowEncoder(embedding_dim=embedding_dim, dropout=0.15)
        self.pos = PositionalEncoding(embedding_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(embedding_dim)
        self.x_head = CoordHead(embedding_dim)
        self.y_head = CoordHead(embedding_dim, hidden=192, dropout=0.15)
        self.z_head = CoordHead(embedding_dim, hidden=160, dropout=0.10)
        self.class_head = nn.Linear(embedding_dim, num_classes)

    def forward(self, mel_seq, gcc_seq):
        batch, seq_len = mel_seq.shape[:2]
        mel = mel_seq.reshape(batch * seq_len, *mel_seq.shape[2:])
        gcc = gcc_seq.reshape(batch * seq_len, *gcc_seq.shape[2:])
        tokens = self.window_encoder(mel, gcc).reshape(batch, seq_len, -1)
        encoded = self.temporal_encoder(self.pos(tokens))
        feat = self.norm(encoded[:, seq_len // 2])
        xyz = torch.cat([self.x_head(feat), self.y_head(feat), self.z_head(feat)], dim=1)
        return xyz, self.class_head(feat)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "parameters": total,
        "trainable_parameters": trainable,
        "model_size_mb_fp32": total * 4 / (1024**2),
    }
