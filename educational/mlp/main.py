import torch
import torch.nn.functional as F
from data_utils import create_dataset_with_context
from pathlib import Path
from educational.mlp.model import MLPModel
from educational.utils import create_mappings, read_words

def main():
    # Define the path to the data file
    file_folder = Path(__file__).resolve().parent
    data_path = file_folder.parent / "names.txt"

    # Read words from the data file and create character mappings
    words = read_words(data_path)
    stoi, itos = create_mappings(words)
    vocab_size = len(itos)
    block_size = 3  # Define the context block size

    # Create datasets for training, development, and testing
    Xtr, Ytr, Xdev, Ydev, Xte, Yte = create_dataset_with_context(data_path, block_size=block_size)

    # Define the MLP model parameters
    n_embd = 10  # Dimensionality of the character embedding vectors
    n_hidden = 200  # Number of neurons in the hidden layer of the MLP

    # Initialize the MLP model
    model = MLPModel(n_embd, n_hidden, vocab_size, block_size)

    # Training

    # Define the learning rate and number of epochs
    lr = 0.1
    num_epochs = 100  # example epoch count

    # Training loop
    for epoch in range(num_epochs):
        # Adjust learning rate if needed
        if epoch == 100000:  # example condition for learning rate change
            lr = 0.01

        # Mini-batch training
        for i in range(Xtr.shape[0] // 32):  # example for batching

            ix = torch.randint(0, Xtr.shape[0], (32,))

            # Forward pass using model's forward method
            logits = model.forward(Xtr[ix])

            # Compute loss
            loss = F.cross_entropy(logits, Ytr[ix])

            # Zero gradients
            for p in model.parameters:
                if p.grad is not None:
                    p.grad.zero_()

            # Backward pass
            loss.backward()

            # Manual parameter update
            with torch.no_grad():
                for p in model.parameters:
                    p -= lr * p.grad

            # Optionally print loss information
            if i % 100 == 0:  # example condition for printing
                print(f"Epoch {epoch}, Step {i}, Loss: {loss.item()}")




if __name__ == "__main__":
    main()
