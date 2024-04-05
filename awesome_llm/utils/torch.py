# Author: Piergiuseppe Mallozzi
# Year: 2024


from pathlib import Path

import torch.nn as nn
from torchinfo import summary
from torchview import draw_graph
from torchviz import make_dot


def save_model_info(
    model: nn.Module, input_tensor: torch.Tensor, folder: Path, id=""
) -> None:
    """
    Save model architecture views, info, and ONNX export to files, with model state management.
    Also, clean up any intermediate files generated in the process.

    Args:
        model (nn.Module): The PyTorch model.
        input_tensor (torch.Tensor): A tensor representative of a single input to the model.
        folder (Path): The directory path where the files will be saved.
    """

    # Ensure the folder exists
    folder.mkdir(parents=True, exist_ok=True)

    # Check model's current mode and set to eval if necessary
    was_training = model.training
    model.eval()

    name = str(model.__class__.__name__)

    try:
        # Export the model to ONNX format
        onnx_path = folder / f"model_{id}_{name}.onnx"
        torch.onnx.export(
            model,  # model being run
            # model input (or a tuple for multiple inputs)
            input_tensor,
            onnx_path,  # where to save the
            # print out a verbose ONNX representation of the
            # model
            verbose=False,
            # store the trained parameter weights inside the
            # model file
            export_params=True,
            # the ONNX version to export the model to
            opset_version=11,
            do_constant_folding=True,
            # whether to execute constant folding for
            # optimization
            # the model's input names
            input_names=["input"],
            output_names=["output"],
            # the model's output names
            dynamic_axes={
                "input": {0: "batch_size"},  # variable length axes
                "output": {0: "batch_size"},
            },
        )
        print(f"Model exported to ONNX format at {onnx_path}")

        try:
            # Save the graphical representation of the model
            output = model(input_tensor)
            parameters = dict(model.named_parameters())
            graph = make_dot(output, parameters, show_attrs=False, show_saved=True)
            torchviz_path = folder / f"torchviz_{id}_{name}"
            graph.render(torchviz_path.stem, format="pdf")
            print(f"Torchviz graph saved to {torchviz_path}.pdf")
        except AttributeError as e:
            print(f"Failed to generate torchviz graph due to an AttributeError: {e}")
        except Exception as e:
            print(
                f"An unexpected error occurred while generating the torchviz graph: {e}"
            )

        # Save the summary of the model
        model_summary = summary(model, input_data=input_tensor, verbose=0)
        summary_path = folder / f"torchinfo_{id}_{name}.txt"
        with open(summary_path, "w") as f:
            f.write(str(model_summary))
        print(f"Model summary saved to {summary_path}")

        model_graph = draw_graph(
            model,
            input_data=input_tensor,
            expand_nested=True,
            hide_inner_tensors=True,
            hide_module_functions=False,
            roll=False,
            depth=20,
        )
        graph = model_graph.visual_graph
        graph.render(folder / f"torchview_{id}_{name}", format="pdf")
        print(f"Torchview graph saved to {folder/'torchview_{id}_{name}.pdf'}")

        intermediate_file = folder / f"torchviz_{id}_{name}"
        if intermediate_file.exists():
            intermediate_file.unlink()
            print(f"Removed intermediate file: {intermediate_file}")

        intermediate_file = folder / f"torchview_{id}_{name}"
        if intermediate_file.exists():
            intermediate_file.unlink()
            print(f"Removed intermediate file:  {intermediate_file}")

    finally:
        # Ensure the model is returned to its original training state
        if was_training:
            model.train()
