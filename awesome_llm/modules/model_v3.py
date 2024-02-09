import torch
import torch.nn as nn
import torch.nn.functional as F

class GPT_v3(nn.Module):
    """
    An advancement in the GPT model series that introduces positional embeddings to the architecture.
    This addition allows the model to understand the sequential order of tokens, which is vital for
    capturing the context and nuances of language. Positional embeddings are combined with token
    embeddings to provide the model with both the content and position information of each token in
    a sequence, enhancing its ability to generate coherent and contextually relevant sequences.

    Compared to GPT_v2, this model now includes a position embedding table that maps each position
    in a sequence to a unique embedding, enabling the model to consider the order of tokens.
    """
    
    def __init__(self, vocab_size: int, n_embd: int, block_size: int, device: torch.device) -> None:
        """
        Initializes the GPT_v3 model with embedding layers for tokens and positions, and a linear
        layer for the language model head.

        Parameters:
        - vocab_size (int): The size of the vocabulary.
        - n_embd (int): The size of each embedding vector.
        - block_size (int): The maximum length of the input sequences.
        - device (torch.device): The device (CPU or GPU) where the model's tensors will be allocated.
        """
        super(GPT_v3, self).__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.device = device

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the model by combining token and positional embeddings before
        applying the linear transformation to produce logits.

        Parameters:
        - indices (torch.Tensor): A tensor of token indices with shape `(batch_size, sequence_length)`.

        Returns:
        - torch.Tensor: The logits with shape `(batch_size, sequence_length, vocab_size)`.

        This method enhances the model's understanding of sequence order by adding positional information
        to the token embeddings, crucial for generating contextually coherent sequences.
        """
        B, T = indices.shape

        # Generate position indices (0 to T-1) on the model's device for compatibility and efficiency. 
        # This ensures all tensors are on the same device, avoiding data transfer delays.
        position_indices = torch.arange(T, device=self.device)

        # Expand position indices to match the batch size (B), creating a (B, T) tensor.
        # This allows each sequence in the batch to have a unique set of position indices,
        # which is crucial for the model to understand the sequence order.
        position_indices = position_indices.expand(B, -1)

        tok_emb = self.token_embedding_table(indices)  # Token embeddings
        pos_emb = self.position_embedding_table(position_indices)  # Positional embeddings
        x = tok_emb + pos_emb  # Combine token and position embeddings
        logits = self.lm_head(x)  # Apply linear layer to combined embeddings
        return logits

    def generate(self, start_indices: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        Generates new tokens based on the given starting sequence, using the model's current understanding
        and prediction capabilities.

        Parameters:
        - start_indices (torch.Tensor): The starting sequence of token indices.
        - max_new_tokens (int): The maximum number of new tokens to generate.

        Returns:
        - torch.Tensor: The extended sequence of token indices after generation.

        This method iteratively generates new tokens by predicting the next token based on the current sequence,
        showcasing the model's ability to extend sequences in a contextually relevant manner.
        """
        indices = start_indices
        for _ in range(max_new_tokens):
            logits = self(indices)  # Obtain logits for the current sequence
            logits = logits[:, -1, :]  # Focus on the logits for the last token in the sequence
            probabilities = F.softmax(logits, dim=-1)  # Compute softmax to get probabilities
            next_index = torch.multinomial(probabilities, num_samples=1)  # Sample the next token
            indices = torch.cat((indices, next_index), dim=1)  # Append the new token to the sequence

        return indices



