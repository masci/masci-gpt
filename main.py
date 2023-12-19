import torch
import torch.nn as nn
from torch.nn import functional as F

# Make the script reproducible
torch.manual_seed(1337)

#
# hyperparameters
#
# Use the GPU if possible
# device = "mps" if torch.backends.mps.is_available() else "cpu"
device = "cpu"
# Maximum context length for predictions
block_size = 32
# How many independent sequences we'll process in parallel. This translates to
# how many rows will be in the final tensor we send for training.
batch_size = 4
# Training iterations
max_iters = 10_000
# Evaluation check point every
eval_interval = 500
eval_iters = 200
# Embedding dimensions
n_embeds = 32


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embeds)
        self.lm_head = nn.Linear(n_embeds, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets have shape (B,T)

        # embeddings and logits have shape (B,T,C) with
        # B being batch size
        # T being block size (aka TIME)
        # and for embeddings
        # C being the size of the embed (n_embeds in our case)
        tok_embeds = self.token_embedding_table(idx)
        # while for logits
        # C being embedding_dim (vocab_size in our case)
        logits = self.lm_head(tok_embeds)

        if targets is None:
            # When this method is called from `generate`, we won't have the targets
            # so we just don't compute the loss in that case
            loss = None
        else:
            # Now we want to compute the loss, problem is cross_entropy expects
            # the input to have shape B,H so let's reshape our tensors
            B, T, H = logits.shape
            logits = logits.view(B * T, H)  # we squash B and T into one dimension
            targets_v = targets.view(B * T)  # we squash targets into a 1-dim array
            loss = F.cross_entropy(logits, targets_v)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


@torch.no_grad()  # Tell torch we won't need backpropagation in this function
def estimate_loss(model, dataset):
    losses = torch.zeros(eval_iters)

    model.eval()
    for k in range(eval_iters):
        X, Y = get_batch(dataset)
        logits, loss = model(X, Y)
        losses[k] = loss.item()
    model.train()

    return losses.mean()


def get_text(fname: str) -> str:
    with open(fname) as f:
        return f.read()


def get_vocabulary(text) -> list[str]:
    chars = sorted(list(set(text)))
    return chars


def get_token_maps(vocabulary):
    stoi = {ch: i for i, ch in enumerate(vocabulary)}
    itos = {i: ch for i, ch in enumerate(vocabulary)}
    return stoi, itos


def encode(input_s: str, stoi: dict[str, int]):
    return [stoi[c] for c in input_s]


def decode(input_t, itos: dict[int, str]) -> str:
    return "".join([itos[i] for i in input_t])


def split_dataset(data):
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data


def get_batch(data):
    """
    Return a batch of data inputs x and a batch of targets y
    """
    # ix is an array of `batch_size` random positions within the dataset
    # don't pick numbers beyond max_index, we want to have full `block_size` samples
    max_index = len(data) - block_size
    # 1-dim array of four numbers
    size = (batch_size,)
    ix = torch.randint(max_index, size)

    # now we get the actual samples of length `block_size`,
    # starting from the random positions in ix
    x = torch.stack([data[i : i + block_size] for i in ix])

    # the targets are the "next value" in the sequence for each input,
    # we just pick the i+1 index
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])

    # Move everything to the GPU
    x, y = x.to(device), y.to(device)
    return x, y


def main():
    # Build the vocabulary
    text = get_text("input.txt")
    vocabulary = get_vocabulary(text)
    stoi, itos = get_token_maps(vocabulary)

    # Build the samples
    data = torch.tensor(encode(text, stoi))
    t_data, v_data = split_dataset(data)
    x, y = get_batch(t_data)

    # Build the model
    m = BigramLanguageModel(len(vocabulary))
    model = m.to(device)  # move it to the GPU

    # # Evaluate loss (at the moment, this should be close to -ln(1/vocab_size))
    # out, loss = m(x, y)
    # print(out.shape)
    # print(loss)

    # Make a prediction. We start by passing the token at position 0, which in our
    # vocabulary corresponds to \n
    init_tok = torch.zeros([1, 1], dtype=torch.long, device=device)
    # # Generate 100 tokens from idx
    # idx = m.generate(init_tok, max_new_tokens=200)
    # # Pick the 0th dimension of the tensor, the one containing the batch
    # idx = idx[0]
    # # Convert to list so we can decode it
    # out = decode(idx.tolist(), itos)
    # print(out)

    # The generation above was garbage, let's train the model

    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for step in range(max_iters):
        # monitor the loss every few steps
        if step % eval_interval == 0:
            print(
                f"Step {step}: train loss {estimate_loss(model, t_data):.4f}, validation loss {estimate_loss(model, v_data):.4f}"
            )

        # Sample a batch of data
        xb, yb = get_batch(t_data)

        # Evaluate the loss
        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    idx = m.generate(init_tok, max_new_tokens=200)[0]
    print(decode(idx.tolist(), itos))


if __name__ == "__main__":
    main()
