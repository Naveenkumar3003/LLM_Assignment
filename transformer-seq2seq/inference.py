import torch
import pickle
import torch.nn.functional as F
from transformer import Transformer
from attention_masks import causal_mask

# ---------------- Load Vocabulary ----------------
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

inv_vocab = {v: k for k, v in vocab.items()}
vocab_size = len(vocab)

# ---------------- Encode / Decode ----------------
def encode(text):
    return [vocab[w] for w in text.split()]

def decode(tokens):
    words = []
    for t in tokens:
        if inv_vocab[t] == "<eos>":
            break
        words.append(inv_vocab[t])
    return " ".join(words)

# ---------------- Load Model ----------------
model = Transformer(vocab_size, 64)
model.load_state_dict(torch.load("model.pth"))
model.eval()

# ---------------- Sampling Function ----------------
def sample_next_token(logits, temperature=0.8, top_k=5):
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)

    top_probs, top_indices = torch.topk(probs, top_k)
    next_token = top_indices[torch.multinomial(top_probs, 1)].item()
    return next_token

# ---------------- Autoregressive Generation ----------------
def generate(src_sentence, max_len=25):
    src_ids = torch.tensor([encode(src_sentence)])
    memory = model.encoder(src_ids)

    generated = [vocab["<sos>"]]

    for _ in range(max_len):
        tgt_ids = torch.tensor([generated])
        mask = causal_mask(tgt_ids.size(1))

        out = model.decoder(tgt_ids, memory, mask)
        logits = out[0, -1]

        next_token = sample_next_token(logits)

        # stop if EOS
        if inv_vocab[next_token] == "<eos>":
            break

        # prevent infinite repetition
        if len(generated) > 2 and next_token == generated[-1]:
            break

        generated.append(next_token)

    return decode(generated[1:])

# ---------------- Test ----------------
print("\n--- MODEL GENERATED OUTPUTS ---\n")

test_inputs = [
    "AI improves healthcare",
    "Transformers process data in parallel",
    "What is self-attention?"
]

for text in test_inputs:
    print("Input :", text)
    print("Output:", generate(text))
    print("-" * 50)
