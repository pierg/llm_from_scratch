import torch
from pathlib import Path
import torch.nn.functional as F

from simple.utils import create_mappings, read_words

# Fixed seed for reproducibility
g = torch.Generator().manual_seed(2147483647)

# Constants for model configuration
NUM_CLASSES = 27
EMBEDDINGS_DIM = 10
CONTEXT_SIZE = 3
L1_SIZE = 200

class MLPModel:
    def __init__(self, filename: Path) -> None:
        """
        Initialize the MLP model with a dataset.

        :param filename: A path to the file containing names.
        """
        self.words = read_words(filename)
        self.stoi, self.itos = create_mappings(self.words)

        # Prepare dataset with context
        self.xsc, self.ysc = self.create_dataset_with_context(CONTEXT_SIZE)

        # Initialize the weights of the MLP
        self.C = torch.rand(NUM_CLASSES, EMBEDDINGS_DIM, generator=g).requires_grad_(True)
        self.W1 = torch.randn((CONTEXT_SIZE * EMBEDDINGS_DIM, L1_SIZE), generator=g).requires_grad_(True)
        self.B1 = torch.randn(L1_SIZE, generator=g).requires_grad_(True)
        self.W2 = torch.randn((L1_SIZE, NUM_CLASSES), generator=g).requires_grad_(True)
        self.B2 = torch.randn(NUM_CLASSES, generator=g).requires_grad_(True)

        # List of model parameters
        self.mlp_parameters = [self.C, self.W1, self.B1, self.W2, self.B2]

    def create_dataset_with_context(self, block_size: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Create a dataset of bigrams from the names.

        :param block_size: Context length, i.e., how many characters to use for prediction.
        :return: A tuple of tensors (xs, ys) representing input and target indices.
        """
        xs = []
        ys = []
        for word in self.words:
            chs = list(word) + ['.']
            context = [self.stoi['.']] * block_size

            for ch in chs:
                if ch in self.stoi:
                    ix = self.stoi[ch]
                    xs.append(context.copy())
                    ys.append(ix)
                    context = context[1:] + [ix]

        return torch.tensor(xs, dtype=torch.int64), torch.tensor(ys, dtype=torch.int64)

    def forward_pass(self, X: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass of the model.

        :param X: The input tensor.
        :return: Probabilities tensor for the next character.
        """
        embeddings = self.C[X]  # (N, CONTEXT_SIZE, EMBEDDINGS_DIM)
        hidden = torch.tanh(embeddings.view(-1, CONTEXT_SIZE * EMBEDDINGS_DIM) @ self.W1 + self.B1)
        logits = hidden @ self.W2 + self.B2
        probs = torch.softmax(logits, dim=1)
        return logits, probs

    
    def train_model(self, use_minibatch: bool = True, batch_size: int = 32, learning_rate: float = 0.01, epochs: int = 1, reg_lambda: float = 0.01) -> None:
        """
        Train the MLP model with options for minibatch training.

        :param use_minibatch: Whether to use minibatch training.
        :param batch_size: Size of each minibatch.
        :param learning_rate: Learning rate for the optimizer.
        :param epochs: Number of training epochs.
        :param reg_lambda: Regularization lambda for L2 regularization.
        """
        optimizer = torch.optim.Adam(self.mlp_parameters, lr=learning_rate)
        loss_fn = torch.nn.CrossEntropyLoss()

        for epoch in range(epochs):
            total_loss = 0

            if use_minibatch:
                num_batches = len(self.xsc) // batch_size
                for _ in range(num_batches):
                    batch_indices = torch.randint(0, self.xsc.shape[0], (batch_size,))
                    logits, _ = self.forward_pass(self.xsc[batch_indices])
                    loss = loss_fn(logits, self.ysc[batch_indices])
                    
                    # L2 Regularization
                    l2_reg = sum(param.pow(2).sum() for param in self.mlp_parameters)
                    loss += reg_lambda * l2_reg

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
            else:
                logits, _ = self.forward_pass(self.xsc)
                loss = loss_fn(logits, self.ysc)
                
                # L2 Regularization
                l2_reg = sum(param.pow(2).sum() for param in self.mlp_parameters)
                loss += reg_lambda * l2_reg

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss = loss.item()

            print(f"Epoch {epoch}, Loss: {total_loss / (num_batches if use_minibatch else 1)}")
