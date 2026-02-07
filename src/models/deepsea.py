import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepSEA(nn.Module):
    """
    DeepSEA implementation (Zhou & Troyanskaya, 2015).
    """

    def __init__(self, input_length=1000, num_tasks=1, reverse_complement=False):
        super().__init__()

        self.reverse_complement = reverse_complement

        # Convolutional layers 
        self.conv1 = nn.Conv1d(4, 320, kernel_size=8, padding=0)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.drop1 = nn.Dropout(0.2)

        self.conv2 = nn.Conv1d(320, 480, kernel_size=8, padding=0)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.drop2 = nn.Dropout(0.2)

        self.conv3 = nn.Conv1d(480, 960, kernel_size=8, padding=0)
        self.drop3 = nn.Dropout(0.5)

        # Calculate output dimensions automatically
        self.flattened_size = self._calculate_flattened_size(input_length)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 925)
        self.fc2 = nn.Linear(925, num_tasks)

        self._initialize_weights()

    def _calculate_flattened_size(self, input_length):
        """Calculate the flattened dimension after convolutions."""
        # Original DeepSEA calculations:
        # L1 = input_length - 7  # after conv1 (kernel=8)
        # L2 = (L1 - 3) // 4     # after pool1 (kernel=4, stride=4)
        # L3 = L2 - 7            # after conv2 (kernel=8)
        # L4 = (L3 - 3) // 4     # after pool2 (kernel=4, stride=4)
        # L5 = L4 - 7            # after conv3 (kernel=8)
        
        # Or compute dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 4, input_length)
            
            # Forward through conv layers
            dummy = F.relu(self.conv1(dummy))      # [1, 320, L-7]
            dummy = self.pool1(dummy)              # [1, 320, (L-7-3)//4 + 1]
            dummy = F.relu(self.conv2(dummy))      # [1, 480, (L-7-3)//4 + 1 - 7]
            dummy = self.pool2(dummy)              # [1, 480, ((L-7-3)//4 + 1 - 7 - 3)//4 + 1]
            dummy = F.relu(self.conv3(dummy))      # [1, 960, ((L-7-3)//4 + 1 - 7 - 3)//4 + 1 - 7]
            
            return dummy.numel()  # 960 * final_length

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # -------------------------------------------------
    # Reverse complement helper
    # -------------------------------------------------
    def reverse_complement_sequence(self, x):
        """Compute reverse complement of one-hot encoded sequence."""
        # x: [B, 4, L]
        # Reverse along sequence dimension
        x = torch.flip(x, dims=[2])
        # Swap complementary bases: A↔T, C↔G
        return x[:, [3, 2, 1, 0], :]

    # -------------------------------------------------
    # Single forward pass
    # -------------------------------------------------
    def forward_single(self, x):
        # Layer 1
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.drop1(x)

        # Layer 2
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.drop2(x)

        # Layer 3
        x = F.relu(self.conv3(x))
        x = self.drop3(x)

        # Fully connected
        x = x.reshape(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    # -------------------------------------------------
    # Forward with optional reverse complement
    # -------------------------------------------------
    def forward(self, x):
        """
        Args:
            x: Tensor [B, 4, L] - one-hot encoded DNA sequences
        Returns:
            logits: Tensor [B] or [B, num_tasks]
        """
        if not self.reverse_complement:
            out = self.forward_single(x)
        else:
            # Average predictions from forward and reverse strands
            out_fwd = self.forward_single(x)
            out_rev = self.forward_single(self.reverse_complement_sequence(x))
            out = 0.5 * (out_fwd + out_rev)

        # Squeeze only if binary classification
        if out.size(1) == 1:
            return out.squeeze(-1)
        return out

