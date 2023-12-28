import torch
import torch.nn as nn
from torch.nn import functional as F

# Make the script reproducible
torch.manual_seed(1337)

#
# hyperparameters
#
# Use the GPU if possible
device = "mps" if torch.backends.mps.is_available() else "cpu"
# device = "cpu"
# Maximum context length for predictions
block_size = 256
# How many independent sequences we'll process in parallel. This translates to
# how many rows will be in the final tensor we send for training.
batch_size = 64
# Training iterations
max_iters = 5_000
# Evaluation check point every
eval_interval = 500
eval_iters = 200
# Embedding dimensions
n_embeds = 384
# Learning rate
learning_rate = 3e-4
# Number of heads
n_heads = 6
# Number of layers
n_layers = 6


class Head(nn.Module):
    """
    One head of self-attention
    """

    def __init__(self, head_size):
        super().__init__()
        # Create the linear layers (or linear projections) that we'll apply to all our nodes
        self.key = nn.Linear(n_embeds, head_size, bias=False)
        self.query = nn.Linear(n_embeds, head_size, bias=False)
        self.value = nn.Linear(n_embeds, head_size, bias=False)
        # 'tril' shouldn't be a parameter of the model, so we store it as a Pytorch "buffer"
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        # k is the vector representing the what a token contains
        k = self.key(x)  # (B,T,C)
        # q is the vector representing what the token is looking for
        q = self.query(x)  # (B,T,C)

        # so far, there was no communication between tokens, this will happen now
        # by computing the dot product between keys and queries

        # compute the attention scores (or affinities). We divide for the square root of C to normalize
        # the scores. This is very important specially at init time, to avoid having un-evenly distributed
        # values, otherwise Softmax will converge to the extremes.
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B,T,C) @ (B,C,T) -> (B,T,T)
        # the masking is what makes this layer a decoder block: no communication is allowed with "future" tokens
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # still (B,T,T)
        wei = F.softmax(wei, dim=-1)  # (B,T,T)
        # perform the weighted aggregations of the values. Value is what the token "declares"
        # as the information it carries
        v = self.value(x)  # (B,T,C)
        out = wei @ v  # (B,T,T) @ (B,T,C) --> (B,T,C)
        return out


class MultiHeadAttention(nn.Module):
    """
    Multiple heads of self-attention in parallel
    """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embeds, n_embeds)  # this is for skip connections

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    """
    Simple linear layer followed by a non-linearity.
    This operates on every single token in isolation.
    """

    def __init__(self, n_embeds) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embeds, 4 * n_embeds),
            nn.ReLU(),
            nn.Linear(
                4 * n_embeds, n_embeds
            ),  # this is to project back after the skip connection
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """
    Transformer block: communication followed by computation
    """

    def __init__(self, n_embeds, num_heads):
        super().__init__()
        head_size = n_embeds // num_heads
        self.sa = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedForward(n_embeds)
        self.ln1 = nn.LayerNorm(n_embeds)
        self.ln2 = nn.LayerNorm(n_embeds)

    def forward(self, x):
        # We implement skip connections (or residual connections). We fork off the computation
        # and project back into the residual pathway via addition. At the beginning of the
        # optimization these take precendence over the transformer blocks speeding up optimizations.
        # then they start to kick in later.
        #
        # Note: in the transformer paper the normalization happens after the transformation,
        # but it's common these days to apply normalization before
        x = x + self.sa(self.ln1(x))  # fork off, communicate, come back
        x = x + self.ffwd(self.ln2(x))  # fork off, compute, come back
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size) -> None:
        super().__init__()
        # This is for encoding the id of the token
        self.token_embedding_table = nn.Embedding(vocab_size, n_embeds)
        # This is for encoding the position of the token in the context
        self.position_embedding_table = nn.Embedding(block_size, n_embeds)
        # Transformer blocks
        self.blocks = nn.Sequential(
            *[Block(n_embeds, n_heads) for _ in range(n_layers)]
        )
        self.ln = nn.LayerNorm(n_embeds)

        # We create the Self Attention multi-heads.
        # We divide the number of embeddings by the number of heads so that when
        # we concatenate at the end we got the original 32 dimensions
        self.sa_heads = MultiHeadAttention(4, n_embeds // 4)
        self.ffwd = FeedForward(n_embeds)
        self.lm_head = nn.Linear(n_embeds, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets have shape (B,T)
        B, T = idx.shape

        # embeddings and logits have shape (B,T,C) with
        # B being batch size
        # T being block size (aka TIME)
        #
        # for the embeddings C is the size of the embed (n_embeds in our case)
        tok_embeds = self.token_embedding_table(idx)
        pos_embeds = self.position_embedding_table(
            torch.arange(T, device=device)  # this is 0, 1, 2, ..., Tmax-1
        )  # shape is (T,C)
        #
        # before decoding the logits, we combine token and position
        x = tok_embeds + pos_embeds  #  torch will get us (B, T, C)
        # # after encoding the tokens and their positions, we feed them into the self-attention head
        # x = self.sa_heads(x)  # (B,T,C)
        # # feed forward
        # x = self.ffwd(x)
        # We replace the two calls above with a call to our transformer blocks
        x = self.blocks(x)
        x = self.ln(x)

        # The output of the self-attention head goes into the decoder language modeling head to create
        # the logits. For the logits, C is the embedding_dim (vocab_size in our case)
        logits = self.lm_head(x)

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
            # crop idx to the last block_size tokens. If idx is larger than block_size,
            # our positional embedding table will run out of scope, because it has embeddings
            # only for tokens up to block_size
            idx_cond = idx[:, -block_size:]
            # Get the predictions
            logits, loss = self(idx_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :]
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append sampled index to the running sequence
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

    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
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
