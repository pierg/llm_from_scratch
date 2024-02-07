import torch
import torch.nn.functional as F
from models.neural_network_model import NeuralNetworkModel
from data.names_data_loader import NamesDataLoader

g = torch.Generator().manual_seed(2147483647)


def generate_name(model, data_loader) -> str:
    """
    Generate a name using the provided model and data_loader.

    :param model: The language model used for name generation.
    :param data_loader: The data loader containing the necessary data and mappings.
    :return: A generated name.
    """
    name = ""
    input_char = "."
    while True:
        probs = model.forward(data_loader.stoi[input_char])
        ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
        output_char = data_loader.itos[ix]
        if output_char == ".":
            break
        name += output_char
        input_char = output_char
    return name



def illustrate_forward_process(model: NeuralNetworkModel, data_loader: NamesDataLoader, word: str):
    if not word:
        print("Empty word provided.")
        return

    # Ensure that all characters are in the mapping
    if not all(ch in data_loader.stoi for ch in word):
        print("Word contains characters not in the mapping.")
        return

    # Convert characters to indices suitable for one-hot encoding
    xs = [data_loader.stoi[ch] for ch in word]

    # One-hot encode the input characters
    xenc = F.one_hot(torch.tensor(xs), num_classes=len(data_loader.stoi)).float()

    # Generate probabilities using the forward pass
    probs = model.forward(xenc)  # Adjust this based on how your model's forward method is implemented

    nlls = torch.zeros(len(word))
    for i in range(len(word)):
        x_idx = xs[i]  # input character index
        y_idx = ys[i]  # label character index
        print('--------')
        print(f'Bigram example {i + 1}: {generator.itos[x_idx]}{generator.itos[y_idx]} (indexes {x_idx},{y_idx})')
        print('Input to the neural net:', x_idx)
        print('Output probabilities from the neural net:', probs[i])
        print('Label (actual next character):', y_idx)
        p = probs[i, y_idx]
        print('Probability assigned by the net to the correct character:', p.item())
        logp = torch.log(p)
        nll = -logp
        print('Negative log likelihood:', nll.item())
        nlls[i] = nll

    print('=========')
    print('Average negative log likelihood, i.e., loss =', nlls.mean().item())



def illustrate_backward_process(model: NeuralNetworkModel, data_loader: NamesDataLoader, word: str):
    if not word:
        print("Empty word provided.")
        return

    # Ensure that all characters are in the mapping
    if not all(ch in data_loader.stoi for ch in word):
        print("Word contains characters not in the mapping.")
        return

    # Convert characters to indices suitable for one-hot encoding
    xs = [data_loader.stoi[ch] for ch in word]

    # One-hot encode the input characters
    xenc = F.one_hot(torch.tensor(xs), num_classes=len(data_loader.stoi)).float()

    # Generate probabilities using the forward pass
    probs = model.forward(xenc)  # Adjust this based on how your model's forward method is implemented

    print("Shape of probs:", probs.shape)

    # Take the probabilities with respect to the actual next word
    # 1. torch.arange(len(xs)): This creates a tensor of indices [0, 1, ..., len(xs)-1].
    #    It's used to select each bigram from the 'probs' tensor. For a word of length N,
    #    there are N+1 bigrams including the start and end tokens.
    # 2. probs[torch.arange(len(xs)), ys]: This selects the predicted probabilities
    #    corresponding to the actual next characters. 'probs' is a matrix where each row
    #    corresponds to a character in 'xs' and each column to a potential next character.
    #    By indexing with [torch.arange(len(xs)), ys], we are effectively picking the
    #    probability of each actual next character from each row.
    pred_probs = probs[torch.arange(len(xs)), ys]
    print("Shape of pred_probs:", pred_probs.shape)

    # Compute the loss (negative log likelihood) for the word
    loss = -torch.log(pred_probs).mean()
    print('Loss:', loss.item())

    # Perform backward pass to compute gradients
    # When loss.backward() is called, PyTorch automatically computes the gradients
    # of the loss with respect to all tensors with requires_grad=True. It does this
    # by traversing the computation graph in reverse, from the loss tensor back to
    # the input tensors, applying the chain rule to compute gradients along the way.
    # This is possible because PyTorch keeps track of the graph of all operations
    # performed on tensors with requires_grad=True, allowing it to automatically
    # differentiate these operations.
    loss.backward()

