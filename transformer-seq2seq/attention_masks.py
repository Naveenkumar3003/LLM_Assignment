import torch

def causal_mask(size):
    return torch.triu(torch.ones(size, size), diagonal=1).bool()
