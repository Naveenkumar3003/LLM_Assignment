import torch
import torch.nn as nn
from attention import SelfAttention

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.attention = SelfAttention(d_model)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, attention = self.attention(x)
        x = self.fc(x)
        return x, attention
