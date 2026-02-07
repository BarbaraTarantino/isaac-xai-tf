import torch
import torch.nn as nn
from .datasets import one_hot_encode_batch

def train_epoch_cnn(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for seqs, y in loader:
        X = one_hot_encode_batch(seqs).to(device)
        y = y.float().to(device)

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total += loss.item()
    return total / len(loader)


def train_epoch_bert(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for seqs, y in loader:
        y = y.float().to(device)

        optimizer.zero_grad()
        logits = model(list(seqs))
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total += loss.item()
    return total / len(loader)


def freeze_model(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
