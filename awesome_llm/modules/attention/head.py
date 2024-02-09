import torch
import torch.nn as nn

from awesome_llm.modules.default.linear import Linear

class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, head_size, n_embd, dropout_probability, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = Linear(n_embd, head_size, bias=False)
        self.value = Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # Register a buffer so that it is not a parameter of the model

        self.dropout = nn.Dropout(dropout_probability)

    def forward(self, x):
        B,T,C = x.shape   # Batch size, block size (Time), vocab size (each token is a vector of size 32)
        k = self.key(x)   # (B,T,C) -> (B,T, head_size)
        q = self.query(x) # (B,T,C) -> (B,T, head_size)
        # Compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5                       # (B, T, head_size) @ (B, head_size, T) = (B, T, T) (T is the block_size)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # Masking all values in wei where tril == 0 with -inf
        wei = F.softmax(wei, dim=-1)                                 # (B, T, T)
        wei = self.dropout(wei)
        # Weighted aggregation of the values
        v = self.value(x) # (B, T, C) -> (B, T, head_size)
        out = wei @ v     # (B, T, T) @ (B, T, head_size) = (B, T, head_size)
        return out
