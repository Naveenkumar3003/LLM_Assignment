import torch
import torch.nn as nn
import torch.optim as optim
from encoder import TransformerEncoder
from dataset import data

# -----------------------------
# Build Vocabulary
# -----------------------------
vocab = {"[MASK]": 0}
idx = 1

for sentence, target in data:
    for word in sentence.split():
        if word not in vocab:
            vocab[word] = idx
            idx += 1
    if target not in vocab:
        vocab[target] = idx
        idx += 1

inv_vocab = {v: k for k, v in vocab.items()}
vocab_size = len(vocab)

# -----------------------------
# Encode sentence
# -----------------------------
def encode(sentence):
    return [vocab[word] for word in sentence.split()]

# -----------------------------
# Model setup
# -----------------------------
model = TransformerEncoder(d_model=64, vocab_size=vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# Training
# -----------------------------
for epoch in range(300):
    total_loss = 0

    for sentence, target_word in data:
        tokens = encode(sentence)
        input_ids = torch.tensor([tokens])

        mask_pos = tokens.index(vocab["[MASK]"])
        target_id = torch.tensor([vocab[target_word]])

        optimizer.zero_grad()
        outputs, _ = model(input_ids)

        loss = criterion(outputs[0][mask_pos].unsqueeze(0), target_id)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

# -----------------------------
# Testing (Reconstruction)
# -----------------------------
print("\n--- Final Reconstructed Outputs ---")
for sentence, _ in data:
    tokens = encode(sentence)
    input_ids = torch.tensor([tokens])
    mask_pos = tokens.index(vocab["[MASK]"])

    with torch.no_grad():
        outputs, _ = model(input_ids)

    predicted_id = torch.argmax(outputs[0][mask_pos]).item()
    predicted_word = inv_vocab[predicted_id]

    print(sentence.replace("[MASK]", predicted_word))
