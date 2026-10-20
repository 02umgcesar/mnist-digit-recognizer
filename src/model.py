import torch.nn as nn

class LinearModel(nn.Module):
    """A single linear layer: 784 pixels in, 10 digit scores out."""

    def __init__(self, num_classes: int = 10):
        """Build the layers. Runs once, when the model is created.

        The weights start out random; training adjusts them.
        """
        super().__init__()  # set up nn.Module's tracking of layers and weights
        self.flatten = nn.Flatten()  # [batch, 1, 28, 28] -> [batch, 784]; has no weights
        self.linear = nn.Linear(28 * 28, num_classes)  # 784 pixels -> num_classes scores via y = W @ x + b

    def forward(self, x):
        """Turn a batch of images into digit scores. Runs on every model(x) call.

        x has shape [batch, 1, 28, 28]; the result has shape [batch, num_classes].
        The highest score in each row is the model's prediction.
        """
        return self.linear(self.flatten(x))

class SimpleCNN(nn.Module):
    """A simple CNN model for MNIST (stub).

    Methods are declared but not implemented yet.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        # layers to be defined
        pass

    def forward(self, x):
        """Forward pass (stub).

        Implementation pending.
        """
        pass


def build_model(num_classes: int = 10) -> nn.Module:
    """Construct and return the model instance (stub)."""
    return LinearModel(num_classes)

