import torch.nn as nn
from encoder import Encoder
from decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.encoder = Encoder(vocab_size, d_model)
        self.decoder = Decoder(vocab_size, d_model)

    def forward(self, src, tgt, mask):
        memory = self.encoder(src)
        return self.decoder(tgt, memory, mask)
