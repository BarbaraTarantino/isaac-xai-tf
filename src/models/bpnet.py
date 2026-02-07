import torch
import torch.nn as nn
import torch.nn.functional as F

class BPNetClassifier(nn.Module):
    """
    BPNet-style model (Avsec et al., 2021)
    """

    def __init__(self, n_filters=64, n_dilations=8):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(4, n_filters, kernel_size=21, padding=10),
            nn.BatchNorm1d(n_filters),
            nn.ReLU(),
        )

        dilations = [1, 2, 4, 8, 16, 32, 64, 128][:n_dilations]
        self.dilated_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    n_filters, n_filters,
                    kernel_size=3,
                    padding=d,
                    dilation=d
                ),
                nn.BatchNorm1d(n_filters),
                nn.ReLU(),
                nn.Dropout(0.2),
            )
            for d in dilations
        ])

        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(n_filters, 1)

    def forward(self, x):
        # x is already (B, 4, L)
        h = self.stem(x)
        for block in self.dilated_convs:
            h = h + block(h)

        h = self.pool(h).squeeze(-1)
        return self.fc(h).squeeze(-1)
