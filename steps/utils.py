import torch
import torch.nn as nn
from pathlib import Path
from torchviz import make_dot
from torchinfo import summary
from torchview import draw_graph


def save_model_info(model: nn.Module, input_tensor: torch.Tensor, folder: Path) -> None:
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
    
    try:
        # Export the model to ONNX format
        onnx_path = folder / "model.onnx"
        torch.onnx.export(model,                                    # model being run
                    input_tensor,                                 # model input (or a tuple for multiple inputs)
                    onnx_path,                                    # where to save the 
                    verbose=False,                                # print out a verbose ONNX representation of the model
                    export_params=True,                           # store the trained parameter weights inside the model file
                    opset_version=11,                             # the ONNX version to export the model to
                    do_constant_folding=True,                     # whether to execute constant folding for optimization
                    input_names = ['input'],                      # the model's input names
                    output_names = ['output'],                    # the model's output names
                    dynamic_axes={'input': {0: 'batch_size'},     # variable length axes
                                    'output': {0: 'batch_size'}})
        print(f"Model exported to ONNX format at {onnx_path}")
        
        try:
            # Save the graphical representation of the model
            output = model(input_tensor)
            parameters = dict(model.named_parameters())
            graph = make_dot(output, parameters, show_attrs=False, show_saved=True)
            torchviz_path = folder / "torchviz"
            graph.render(torchviz_path.stem, format="pdf")
            print(f"Torchviz graph saved to {torchviz_path}.pdf")
        except AttributeError as e:
            print(f"Failed to generate torchviz graph due to an AttributeError: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while generating the torchviz graph: {e}")

        
        # Save the summary of the model
        model_summary = summary(model, input_data=input_tensor, verbose=0)
        summary_path = folder / "torchinfo.txt"
        with open(summary_path, "w") as f:
            f.write(str(model_summary))
        print(f"Model summary saved to {summary_path}")
        
        # Use draw_graph for detailed visualization
        print("\nDrew Graph")
        model_graph = draw_graph(model, 
                                input_data=input_tensor, 
                                expand_nested=True, 
                                hide_inner_tensors=True,
                                hide_module_functions=False,
                                roll=False,
                                depth=20)
        graph = model_graph.visual_graph
        graph.render(folder / 'torchview', format='pdf')
        print(f"Torchview graph saved to {folder/'torchview.pdf'}")

        intermediate_file = folder / "torchviz"
        if intermediate_file.exists():
            intermediate_file.unlink()
            print(f"Removed intermediate file: {intermediate_file}")

        intermediate_file = folder / "torchview"
        if intermediate_file.exists():
            intermediate_file.unlink()
            print(f"Removed intermediate file:  {intermediate_file}")

    finally:
        # Ensure the model is returned to its original training state
        if was_training:
            model.train()
