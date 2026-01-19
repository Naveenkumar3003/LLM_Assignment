import torch
import torch.nn as nn
import pickle
from transformer import Transformer
from dataset import pairs
from attention_masks import causal_mask

# ---------------- Vocabulary ----------------
vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2}
idx = 3

for src, tgt in pairs:
    for w in (src + " " + tgt).split():
        if w not in vocab:
            vocab[w] = idx
            idx += 1

vocab_size = len(vocab)

# ---------------- Encoding ----------------
def encode_src(text):
    return [vocab[w] for w in text.split()]

def encode_tgt(text):
    return [vocab["<sos>"]] + [vocab[w] for w in text.split()] + [vocab["<eos>"]]

# ---------------- Model ----------------
model = Transformer(vocab_size, 64)
loss_fn = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ---------------- Training ----------------
for epoch in range(300):
    total_loss = 0
    for src, tgt in pairs:
        src_ids = torch.tensor([encode_src(src)])
        tgt_ids = torch.tensor([encode_tgt(tgt)])

        decoder_input = tgt_ids[:, :-1]
        decoder_target = tgt_ids[:, 1:]

        mask = causal_mask(decoder_input.size(1))
        output = model(src_ids, decoder_input, mask)

        loss = loss_fn(
            output.reshape(-1, vocab_size),
            decoder_target.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 50 == 0:
        print(f"Epoch {epoch} | Loss: {total_loss:.4f}")

# ---------------- Save ----------------
torch.save(model.state_dict(), "model.pth")
with open("vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)

print("Model and vocabulary saved successfully")
