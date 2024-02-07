import numpy as np
import torch
from awesome_llm.modules.base import SimpleModule
from awesome_llm.modules.embedding import Embedding
from awesome_llm.utils.math import cross_entropy_loss, multinomial, softmax


class BigramLM(SimpleModule):
    """
    A Bigram Language Model using a custom embedding layer.
    """
    def __init__(self, vocab_size: int):
        """
        Initializes the BigramLM model.

        :param vocab_size: The size of the vocabulary.
        """
        super().__init__()
        # Embedding layer initialization. 
        # Each token in the vocabulary is represented by a vector of size 'vocab_size'.
        # The embedding layer works as a look-up table with dimensions (vocab_size, vocab_size).
        self.embed = Embedding(vocab_size, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass on the model.

        :param idx: Indices representing the input batch.
        :return: Logits as a result of the forward pass.
        """

        # Embedding input indices to obtain logits.
        # The resulting shape is (batch_size, block_size, vocab_size), denoted as (B, T, C).
        # B: batch size, T: sequence length (time), C: embedding size (channels).
        logits = self.embed(idx)

        return logits


    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        Generates new tokens based on the last token of a sequence.

        :param idx: Starting sequence of indices.
        :param max_new_tokens: Maximum number of new tokens to generate.
        :return: Extended sequence of indices.
        """       
        for _ in range(max_new_tokens):
            logits = self(idx)                               # Forward pass with the current sequence
            logits = logits[:, -1, :]                           # Focus on the last token from the logits
            probs = torch.nn.functional.softmax(logits, dim=-1) # Probability distribution for the next token
            idx_next = torch.multinomial(probs, num_samples=1)  # Sample the next token
            idx = torch.cat((idx, idx_next), dim=1)             # Add the new token to the sequence
        return idx




