import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepBind(nn.Module):
    """
    DeepBind-style model (Alipanahi et al., 2015).
    """

    def __init__(
        self,
        num_motifs=16,
        motif_len=24,
        pool_type="max",
        dropout=0.5,
        reverse_complement=False,
        use_hidden=True,
        hidden_size=32,
    ):
        super().__init__()

        self.pool_type = pool_type
        self.reverse_complement = reverse_complement
        self.use_hidden = use_hidden

        # -------------------------------------------------
        # Motif detectors 
        # -------------------------------------------------
        self.conv = nn.Conv1d(
            in_channels=4,
            out_channels=num_motifs,
            kernel_size=motif_len,
            padding=0
        )

        nn.init.normal_(self.conv.weight, mean=0.0, std=0.01)
        with torch.no_grad():
            self.conv.bias.data = -torch.abs(self.conv.bias.data)

        # -------------------------------------------------
        # Pooling
        # -------------------------------------------------
        if pool_type == "max":
            self.pool = nn.AdaptiveMaxPool1d(1)
        elif pool_type == "maxavg":
            self.max_pool = nn.AdaptiveMaxPool1d(1)
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
        else:
            self.pool = nn.AdaptiveAvgPool1d(1)

        # -------------------------------------------------
        # Dropout 
        # -------------------------------------------------
        self.dropout = nn.Dropout(p=dropout)

        # -------------------------------------------------
        # Hidden / output
        # -------------------------------------------------
        pooled_dim = num_motifs if pool_type != "maxavg" else 2 * num_motifs

        if use_hidden:
            self.hidden = nn.Linear(pooled_dim, hidden_size)
            self.output = nn.Linear(hidden_size, 1)
            nn.init.normal_(self.hidden.weight, mean=0.0, std=0.3)
            nn.init.zeros_(self.hidden.bias)
        else:
            self.hidden = None
            self.output = nn.Linear(pooled_dim, 1)

        nn.init.normal_(self.output.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.output.bias)

    # -------------------------------------------------
    # Reverse complement (sequence-level)
    # -------------------------------------------------
    def reverse_complement_sequence(self, x):
        # x: [B, 4, L]
        x = torch.flip(x, dims=[2])
        return x[:, [3, 2, 1, 0], :]

    # -------------------------------------------------
    # Single forward
    # -------------------------------------------------
    def forward_single(self, x):
        x = F.relu(self.conv(x))

        if self.pool_type == "maxavg":
            max_p = self.max_pool(x).squeeze(-1)
            avg_p = self.avg_pool(x).squeeze(-1)
            x = torch.cat([max_p, avg_p], dim=1)
        else:
            x = self.pool(x).squeeze(-1)

        x = self.dropout(x)

        if self.use_hidden:
            x = F.relu(self.hidden(x))

        return self.output(x)

    # -------------------------------------------------
    # Forward
    # -------------------------------------------------
    def forward(self, x):
        if not self.reverse_complement:
            out = self.forward_single(x)
        else:
            out_fwd = self.forward_single(x)
            out_rev = self.forward_single(self.reverse_complement_sequence(x))
            # strand-invariant, smooth
            out = 0.5 * (out_fwd + out_rev)

        return out.squeeze(-1)

