import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from pathlib import Path
import torch.nn.functional as F

from educational.utils import create_mappings, read_words


output_folder = Path(__file__).parent / "output"
if not output_folder.exists():
    output_folder.mkdir()

class BigramNameGenerator:
    """
    A class to generate names using a bigram model.
    """

    def __init__(self, filename: Path) -> None:
        """
        Initializes the bigram name generator with a dataset.

        :param filename: A path to the file containing names.
        """
        self.words = read_words(filename)
        self.bigrams = self.bigram_counts()
        self.stoi, self.itos = create_mappings(self.words)

        # Table of Probabilities
        self.N = self.calculate_bigram_counts_matrix()
        self.P = self.calculate_bigram_probabilities_matrix(self.N)

        # One Layer Neural Network
        self.xs, self.ys = self.create_dataset()
        self.W = self.initialize_network()


    def bigram_counts(self) -> dict:
        """
        Counts bigrams in the words list.

        :return: A dictionary of bigram counts.
        """
        b = {}
        for word in self.words:
            w = ["<S>"] + list(word) + ["<E>"]
            for ch1, ch2 in zip(w, w[1:]):
                bigram = (ch1, ch2)
                b[bigram] = b.get(bigram, 0) + 1
        return dict(sorted(b.items(), key=lambda x: x[1], reverse=True))

    def calculate_bigram_counts_matrix(self) -> torch.Tensor:
        """
        Calculates a matrix of bigram counts.

        :return: A matrix of bigram counts.
        """
        N = torch.zeros((len(self.stoi), len(self.stoi)), dtype=torch.int32)
        for word in self.words:
            w = ["."] + list(word) + ["."]
            for ch1, ch2 in zip(w, w[1:]):
                ix1, ix2 = self.stoi[ch1], self.stoi[ch2]
                N[ix1, ix2] += 1
        return N

    def calculate_bigram_probabilities_matrix(self, count_matrix: torch.Tensor) -> torch.Tensor:
        """
        Converts a count matrix to a probability matrix with Laplace smoothing.

        :param count_matrix: The bigram counts matrix.
        :return: A matrix of smoothed bigram probabilities.
        """
        # Apply Laplace smoothing: Add 1 to each count
        smoothed_count_matrix = count_matrix + 1
        # Calculate probabilities
        P = smoothed_count_matrix.float()
        P = P / P.sum(axis=1, keepdim=True)
        return P

    def save_plot(self, data: torch.Tensor, filename: str, plot_type: str = "bigram") -> None:
        """
        Saves a plot of the bigram data.

        :param data: The data to be plotted.
        :param filename: The filename for the saved plot.
        :param plot_type: The type of plot to save ("bigram" or "heatmap").
        """
        if plot_type == "bigram":
            plt.figure(figsize=(20, 3))
            plt.bar([a+b for a, b in self.bigrams.keys()][:50], list(self.bigrams.values())[:50])
            plt.xticks(rotation=45)
            plt.tight_layout()
        elif plot_type == "heatmap":
            plt.figure(figsize=(20, 20))
            sns.heatmap(data, xticklabels=list(self.stoi.keys()), yticklabels=list(self.stoi.keys()), annot=True, fmt="g")
            plt.title("Bigram counts heatmap")
            plt.xlabel("second letter")
            plt.ylabel("first letter")
        plt.savefig(output_folder / filename)

    def generate_names_tabular(self, num_names: int = 10) -> list:
        """
        Generates a list of names using the tabular bigram model.

        :param num_names: The number of names to generate.
        :return: A list of generated names.
        """
        g = torch.Generator().manual_seed(2147483647)
        names = []
        for _ in range(num_names):
            ix = 0
            char = ""
            string = ""
            while char != ".":
                p = self.P[ix]
                ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
                char = self.itos[ix]
                string += char
            names.append(string[:-1])  # Remove the "." at the end
        return names

    def generate_names_neural_network(self, num_names: int = 5) -> list:
        """
        Generates a list of names using the neural network model.

        :param num_names: The number of names to generate.
        :return: A list of generated names.
        """
        g = torch.Generator().manual_seed(2147483647)
        names = []
        for _ in range(num_names):
            out = []
            ix = 0
            while True:
                xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
                logits = xenc @ self.W
                counts = logits.exp()
                p = counts / counts.sum(1, keepdims=True)

                ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
                out.append(self.itos[ix])
                if ix == 0:
                    break
            names.append(''.join(out[:-1]))  # Remove the "." at the end
        return names

    
    def calculate_neg_log_likelihood(self, word: str) -> float:
        """
        Calculates the negative log likelihood of a given word based on the bigram model.

        Maximizing the data likelihood which is equivalent to maximizing the log likelihood (since the logarithm is a monotonic function). 
        This in turn is equivalent to minimizing the negative log likelihood.

        :param word: The word for which to calculate the log likelihood.
        :return: The negative log likelihood of the word.
        """
        if not word:
            # Log likelihood of an empty string is not defined
            return float('inf')

        # Prepend and append '.' to denote start and end of word
        w = ["."] + list(word) + ["."]
        # Generate bigram indices while ensuring characters are in the mapping
        bigram_indices = [(self.stoi[ch1], self.stoi[ch2]) for ch1, ch2 in zip(w, w[1:]) 
                          if ch1 in self.stoi and ch2 in self.stoi]
        
        if not bigram_indices:
            # Return inf if any character/bigram is not in the mapping
            return float('inf')

        # Convert bigram indices to tensor for PyTorch operations
        bigram_tensor_indices = torch.tensor(bigram_indices, dtype=torch.int)

        # Extract probabilities for the bigrams from the probability matrix
        bigram_probs = self.P[bigram_tensor_indices[:, 0], bigram_tensor_indices[:, 1]]

        # Print the bigram probabilities for diagnostic purposes
        for idx, (ch1, ch2) in enumerate(zip(w, w[1:])):
            print(f"Bigram '{ch1}{ch2}': Probability = {bigram_probs[idx].item():.4f}")

        # Calculate and sum the log of the probabilities
        log_likelihood = torch.log(bigram_probs).sum().item()
        print(f"Total Log Likelihood for '{word}': {log_likelihood:.4f}")

        # Return the negative log likelihood divided by the number of bigrams
        return -log_likelihood / len(bigram_indices)


    def create_dataset(self):
        """
        Creates a dataset of bigrams from the words.

        :return: A tuple of tensors (xs, ys) representing input and target indices.
        """
        xs = []
        ys = []
        for word in self.words:
            # Add start and end tokens to each word
            chs = ['.'] + list(word) + ['.']
            for ch1, ch2 in zip(chs, chs[1:]):
                if ch1 in self.stoi and ch2 in self.stoi:
                    ix1 = self.stoi[ch1]
                    ix2 = self.stoi[ch2]
                    xs.append(ix1)
                    ys.append(ix2)

        # Convert lists to PyTorch tensors
        xs_tensor = torch.tensor(xs, dtype=torch.int64)
        ys_tensor = torch.tensor(ys, dtype=torch.int64)

        return xs_tensor, ys_tensor
    

    def initialize_network(self):
        """
        Initializes the weights of a simple one-layer neural network.
        The network consists of 27 neurons, each receiving 27 inputs. 
        The weights are randomly initialized. If weights are initialized to zero
        instead of randomly, the logits will be zero, and counts will be one.
        This is equivalent to sampling on a uniform distribution, akin to label smoothing.
        
        :return: Randomly initialized weights tensor with gradients computation enabled.
        """
        g = torch.Generator().manual_seed(2147483647)
        W = torch.randn((27, 27), generator=g)
        return W.requires_grad_(True)

    def forward_pass(self, xenc):
        """
        Performs a forward pass on the neural network.

        :param xenc: One-hot encoded input tensor. This tensor represents the input characters
                    in a one-hot encoded format, which is suitable for processing by the neural network.

        :return: A tensor representing the probabilities of the next character.
        """

        # Multiply the input encoding by the weight matrix (W) to get logits.
        # Logits are essentially the raw predictions made by the network before applying any activation function.
        logits = xenc @ self.W  

        # Apply the exponential function to the logits to get the counts. This step converts the logits
        # (which can be any real number) into positive counts and is a part of the softmax function.
        counts = logits.exp()  

        # Normalize the counts to get probabilities. The softmax function is implemented here by dividing
        # each count by the sum of all counts. This ensures that the output is a valid probability distribution.
        probs = counts / counts.sum(1, keepdims=True)  

        return probs


    def train_model(self, learning_rate=0.01, epochs=1, reg_lambda=0.01):
        """
        Trains the model using gradient descent.

        :param learning_rate: Learning rate for gradient descent. Determines the step size at each iteration while moving toward a minimum of a loss function.
        :param epochs: Number of training iterations. Each epoch is a complete pass over the entire training dataset.
        :param reg_lambda: Regularization parameter. Helps to avoid overfitting by penalizing large weights.

        The regularization term (self.W ** 2) pushes W towards zero, leading to a uniform distribution.
        This process is a form of regularization akin to Laplace smoothing in the tabular form,
        balancing between matching the data-driven probabilities and maintaining a uniform distribution.
        """

        # Re-initialize network weights for training
        self.W = self.initialize_network()

        for epoch in range(epochs):
            # Convert input indices to one-hot encoded tensors, which are suitable for processing by the neural network.
            xenc = F.one_hot(self.xs, num_classes=27).float()

            # Forward pass: compute predicted probabilities using the current state of the network
            probs = self.forward_pass(xenc)

            # Step 1: Gather the predicted probabilities for the actual next characters
            # torch.arange(len(self.xs)) creates a tensor [0, 1, 2, ..., len(self.xs)-1], 
            # which is used to select the corresponding row in 'probs'.
            # self.ys contains the actual next character indices, which is used to select
            # the corresponding column in 'probs'. This results in a tensor where each element
            # is the predicted probability of the actual next character.
            correct_probs = probs[torch.arange(len(self.xs)), self.ys]

            # Step 2: Compute the negative log likelihood
            # torch.log computes the logarithm of each element in 'correct_probs'.
            # Since we are calculating the negative log likelihood, we negate the result.
            neg_log_likelihood = -torch.log(correct_probs)

            # Step 3: Calculate the mean loss
            # The mean() function calculates the average of all elements in 'neg_log_likelihood'.
            # This gives us a single scalar value representing the average loss across all predictions.
            mean_loss = neg_log_likelihood.mean()

            # Step 4: Apply regularization to the loss
            # Regularization helps to prevent the model from overfitting. It is calculated as the
            # mean of the squared weights, multiplied by the regularization lambda.
            reg_loss = reg_lambda * (self.W ** 2).mean()

            # Step 5: Combine mean loss and regularization loss
            loss = mean_loss + reg_loss

            # Print the loss for this epoch to monitor the training progress
            print(f'Epoch {epoch}, Loss: {loss.item()}')

            # Backward pass: compute the gradient of the loss with respect to the weights (W).
            self.W.grad = None  # Clear any existing gradients
            loss.backward()  # Compute new gradients

            # Update weights using gradient descent. The learning rate determines the size of the update step.
            with torch.no_grad():
                self.W -= learning_rate * self.W.grad
