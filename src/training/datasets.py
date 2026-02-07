import torch
from torch.utils.data import Dataset
import numpy as np

BASE2IDX = {"A": 0, "C": 1, "G": 2, "T": 3}

def one_hot_encode_batch(seqs):
    L = len(seqs[0])
    X = np.zeros((len(seqs), 4, L), dtype=np.float32)
    for i, s in enumerate(seqs):
        for j, b in enumerate(s):
            X[i, BASE2IDX[b], j] = 1.0
    return torch.tensor(X)

class DNACNNDataset(Dataset):
    """
    Dataset for CNN-based models (DeepBind, DeepSEA)
    """
    def __init__(self, df, view):
        self.seqs = df[view].values
        self.labels = df["label"].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.seqs[idx], self.labels[idx]

