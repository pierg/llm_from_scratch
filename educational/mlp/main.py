import torch
import torch.nn.functional as F
from data_utils import create_dataset_with_context
from pathlib import Path
from educational.mlp.model import MLPModel
from educational.utils import create_mappings, read_words
import matplotlib.pyplot as plt




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
    num_epochs = 5  # example epoch count
    batch_size = 32  # example batch size

    loss_history = []


    # Training loop
    for epoch in range(num_epochs):
        # Adjust learning rate if needed
        # This condition checks if the current epoch number has reached a certain threshold (100000 in this example),
        # and if so, it reduces the learning rate to 0.01. This technique is often used in training neural networks
        # to fine-tune the model as training progresses.
        if epoch == 10:  # example condition for learning rate change
            lr = 0.01

        # Iterate over the training data in mini-batches
        for i in range(0, Xtr.shape[0], batch_size):  # Batching in steps of 'batch_size'
            # Mini-batch construct
            # This line randomly selects 'batch_size' indices from the range 0 to the size of the training data.
            # These indices will be used to select a subset of the training data (Xtr) and the corresponding labels (Ytr)
            # to form a mini-batch for training.
            ix = torch.randint(0, Xtr.shape[0], (batch_size,))

            # Forward pass using model's forward method
            # The model takes the selected mini-batch of data and performs a forward pass to compute the logits (raw predictions).
            logits = model.forward(Xtr[ix])

            # Compute loss
            # The cross-entropy loss is calculated between the logits and the true labels of the mini-batch.
            # This loss quantifies how well the model is performing on the current mini-batch.
            loss = F.cross_entropy(logits, Ytr[ix])

            # Zero gradients
            # Before computing the backward pass, we zero out the gradients of all parameters.
            # This is necessary because gradients accumulate by default in PyTorch.
            for p in model.parameters:
                if p.grad is not None:
                    p.grad.zero_()

            # Backward pass
            # This computes the gradient of the loss with respect to all parameters in the model.
            loss.backward()

            # Manual parameter update
            # We update the model parameters using a simple gradient descent step.
            # The torch.no_grad() context is used to ensure that these operations are not tracked
            # for gradient computation.
            with torch.no_grad():
                for p in model.parameters:
                    p -= lr * p.grad

            if i % 100 == 0:
                print(f"Epoch {epoch}, Step {i // 32}, Loss: {loss.item()}")


            loss_history.append(loss.item())

    # After training, plot the loss
    plt.plot(loss_history)
    plt.title('Training Loss Over Time')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.savefig('loss_plot.png')


    # Visualize Embeddings
    # Assuming `itos` is the mapping from indices to characters
    characters = [itos[i] for i in range(vocab_size)]

    # Extract the first two dimensions of the embedding matrix
    embeddings = model.C.detach().numpy()[:, :2]

    # Plot each character in the 2D space
    plt.figure(figsize=(10, 8))
    for i, char in enumerate(characters):
        plt.scatter(embeddings[i, 0], embeddings[i, 1])
        plt.text(embeddings[i, 0], embeddings[i, 1], char)

    plt.title('Character Embeddings Visualized in 2D')
    plt.xlabel('Dimension 0')
    plt.ylabel('Dimension 1')
    plt.savefig('embeddings.png')

    def sample_name(context, max_length=20):
        # Initialize the input tensor with the index of the starting character
        input_context = [stoi[char] for char in context]
        output_name = []

        for _ in range(max_length - 1):
            logits = model.forward(torch.tensor(input_context, dtype=torch.long))
            probs = torch.softmax(logits, dim=1)
            next_char_ix = torch.multinomial(probs, 1).item()
            next_char = itos[next_char_ix]
            output_name += next_char

            # Stop if the next character is a period, which we'll use as a termination symbol
            if next_char == '.':
                break

            input_context = input_context[1:] + [next_char_ix]
        return "".join(output_name)

    # Example: Generate 20 names
    for _ in range(20):
        context = ["."] * block_size # initialize with all ...
        generated_name = sample_name(context)
        print(f"Generated Name: {generated_name}")





if __name__ == "__main__":
    main()
