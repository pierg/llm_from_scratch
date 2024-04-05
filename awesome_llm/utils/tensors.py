# Author: Piergiuseppe Mallozzi
# Year: 2024


def numpy_print_batch_info(batch_x, batch_y):
    print("Batch X shape:", batch_x.shape)
    print("Batch Y shape:", batch_y.shape)
    print("Sample entries from Batch X:\n", batch_x[:5])
    print("Sample entries from Batch Y:\n", batch_y[:5])


def print_batch_info(input_tensor: torch.Tensor, target_tensor: torch.Tensor):
    """
    Prints information about the batch including the shape of the input and target tensors,
    and details of the first batch.

    :param input_tensor: The input tensor (x) from get_batch.
    :param target_tensor: The target tensor (y) from get_batch.
    """
    # Print shapes of the input and target tensors
    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Target tensor shape: {target_tensor.shape}\n")

    # Determine the batch size and block size from the input tensor
    batch_size, block_size = input_tensor.shape

    # Print details of the first batch
    print("Details of the first batch:")
    for b in range(batch_size):  # Iterate over each sequence in the batch
        for t in range(block_size):  # Iterate over each token in the sequence
            # Extract the context (prompt) from the input tensor
            context = input_tensor[b, : t + 1]
            # Extract the corresponding target token
            target = target_tensor[b, t]
            # Print the prompt and the target
            print(f"When the prompt is {context.tolist()}, predict {target.item()}")


def pretty_print_tensor(
    tensor: torch.Tensor, name: str = "Tensor", num_entries: int = 2
):
    """
    Pretty prints information about a PyTorch tensor.

    :param tensor: The tensor to be printed.
    :param num_entries: The number of entries from the tensor to display (default is 10).
    """
    print("-------------------")
    print(f"{name} Information:")
    # Print the shape and datatype of the tensor
    print(f"Shape: {tensor.shape}\tDatatype: {tensor.dtype}")
    # Print the first few entries of the tensor followed by "..."
    print(f"Data: {tensor.tolist()[:num_entries]}...")
