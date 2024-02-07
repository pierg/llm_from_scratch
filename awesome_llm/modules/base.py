import torch

class SimpleModule:
    """
    Base class for all modules in the framework.
    """
    def __init__(self):
        self._parameters = {}

    def parameters(self):
        """
        Yield all parameters of the module.
        """
        for param in self._parameters.values():
            yield param

    def add_parameter(self, name: str, value: torch.Tensor):
        """
        Add a parameter to the module.

        :param name: Name of the parameter.
        :param value: Parameter value (torch tensor).
        """
        self._parameters[name] = value

    def forward(self, *input):
        """
        Forward pass to be implemented by the subclass.
        """
        raise NotImplementedError

    def __call__(self, *input):
        """
        Makes the module callable and automatically calls the forward method.
        """
        return self.forward(*input)