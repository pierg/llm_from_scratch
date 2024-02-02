import torch
g = torch.Generator().manual_seed(2147483647)
class MLPModel:
    """
    A simple MLP model for token prediction using character embeddings.

    Attributes:
        C (torch.Tensor): Character embedding matrix.
        W1 (torch.Tensor): Weight matrix for the first layer.
        b1 (torch.Tensor): Bias vector for the first layer.
        W2 (torch.Tensor): Weight matrix for the second layer.
        b2 (torch.Tensor): Bias vector for the second layer.
    """

    def __init__(self, n_embd: int, n_hidden: int, vocab_size: int, block_size: int) -> None:
        """
        Initializes the MLP model with random weights.

        Args:
            n_embd (int): Dimensionality of character embedding vectors.
            n_hidden (int): Number of neurons in the hidden layer.
            vocab_size (int): Size of the vocabulary.
            block_size (int): Size of the context block for prediction.
        """
        g = torch.Generator().manual_seed(2147483647)

        self.n_embd = n_embd
        self.block_size = block_size

        self.C  = torch.randn((vocab_size, n_embd), generator=g)
        self.W1 = torch.randn((n_embd * block_size, n_hidden), generator=g)
        self.b1 = torch.randn(n_hidden, generator=g)
        self.W2 = torch.randn((n_hidden, vocab_size), generator=g)
        self.b2 = torch.randn(vocab_size, generator=g)

        self.parameters = [self.C, self.W1, self.b1, self.W2, self.b2]
        print(f"Total number of parameters: {sum(p.nelement() for p in self.parameters)}")
        for p in self.parameters:
            p.requires_grad = True

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass of the model.

        Args:
            X (torch.Tensor): The input tensor of token indices.

        Returns:
            torch.Tensor: Logits tensor representing the unnormalized log probabilities of the next character.
        """
        embeddings = self.C[X]  
        hidden = torch.tanh(embeddings.view(-1, self.n_embd * self.block_size) @ self.W1 + self.b1)
        logits = hidden @ self.W2 + self.b2
        return logits
