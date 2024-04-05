# Author: Piergiuseppe Mallozzi
# Year: 2024


from typing import Callable

from awesome_llm.data.base_data import BatchGenerator
from awesome_llm.optimizers.base import Optimizer


class Trainer:
    """
    Trainer class for training a SimpleModule model using a specified optimizer.
    Handles the training loop, loss calculation, and evaluation.
    """

    import torch.nn as nn

    def __init__(
        self,
        data: BatchGenerator,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: Callable,
    ):
        """
        Initializes the Trainer with data, model, optimizer, and loss function.
        """
        self.data = data
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train(self, batch_size: int, max_iters: int, eval_interval: int):
        """
        Trains the model for a specified number of iterations and batch size.
        Evaluates the model at regular intervals.
        """
        for iter in range(max_iters):
            # Get a batch of training data
            xb, yb = self.data.get_batch("train", batch_size=batch_size)

            # Forward pass: compute predictions from model
            logits = self.model(xb)

            # Reshaping logits and targets for loss computation
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = yb.view(B * T)

            # Calculate loss using the specified loss function
            loss = self.loss_fn(logits_flat, targets_flat)

            # Backward pass: compute gradients and update model parameters
            self.optimizer.zero_grad()  # Clear existing gradients
            loss.backward()  # Compute new gradients
            self.optimizer.step()  # Update model parameters

            # Periodically evaluate the model and print the losses
            if iter % eval_interval == 0:
                losses = self.evaluate_loss(batch_size)
                print(
                    f'Iter {iter:4d} | Train Loss {losses["train"]:6.4f} | Val Loss {losses["val"]:6.4f}'
                )

    @torch.no_grad()  # Disable gradient calculation for this function
    def evaluate_loss(self, batch_size: int) -> dict:
        """
        Evaluates the model's performance on the train and validation sets.
        Returns a dictionary of average losses for each set.
        """
        out = {}

        self.model.eval()  # Set model to evaluation mode
        eval_iters = 200

        # Loop over both train and validation sets
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)

            for k in range(eval_iters):
                # Get a batch of data for the current split
                xb, yb = self.data.get_batch(split, batch_size=batch_size)

                # Forward pass to compute logits
                logits = self.model(xb)
                B, T, C = logits.shape
                logits_flat = logits.view(B * T, C)
                targets_flat = yb.view(B * T)

                # Calculate loss for the current batch
                loss = self.loss_fn(logits_flat, targets_flat)
                losses[k] = loss.item()

            # Store the mean loss for the current split
            out[split] = losses.mean()

        self.model.train()  # Set model back to training mode
        return out
