import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.self_attn = nn.MultiheadAttention(d_model, 1)
        self.cross_attn = nn.MultiheadAttention(d_model, 1)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x, memory, mask):
        x = self.embedding(x).permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)

        self_attn_out, _ = self.self_attn(x, x, x, attn_mask=mask)
        cross_attn_out, _ = self.cross_attn(self_attn_out, memory, memory)

        out = self.fc(cross_attn_out)
        return out.permute(1, 0, 2)
