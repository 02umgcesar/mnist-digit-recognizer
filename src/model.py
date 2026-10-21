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
    """Two convolution blocks that find patterns, then one linear layer that scores the digits."""

    def __init__(self, num_classes: int = 10):
        """Build the layers. Runs once, when the model is created.

        The filters and weights start out random; training adjusts them.
        """
        super().__init__()  # set up nn.Module's tracking of layers and weights

        # Block 1: 16 filters of 3x3 look for small patterns (edges, curves) anywhere in the image.
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)   # [batch, 1, 28, 28] -> [batch, 16, 28, 28]
        # Block 2: 32 filters combine block 1's patterns into bigger ones (corners, loops).
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # [batch, 16, 14, 14] -> [batch, 32, 14, 14]
        # Block 3: 64 filters combine block 2's patterns into even larger ones.
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # [batch, 32, 7, 7] -> [batch, 64, 7, 7]

        self.relu = nn.ReLU()        # keeps positive values, turns negative ones into 0; has no weights
        self.pool = nn.MaxPool2d(2)  # keeps the largest value in each 2x2 square, halving height and width; no weights

        self.flatten = nn.Flatten()                       # [batch, 64, 3, 3] -> [batch, 576]
        self.linear = nn.Linear(64 * 3 * 3, num_classes)  # 576 pattern values -> num_classes scores via y = W @ x + b

    def forward(self, x):
        """Turn a batch of images into digit scores. Runs on every model(x) call.

        x has shape [batch, 1, 28, 28]; the result has shape [batch, num_classes].
        The highest score in each row is the model's prediction.
        """
        x = self.pool(self.relu(self.conv1(x)))  # [batch, 16, 14, 14]
        x = self.pool(self.relu(self.conv2(x)))  # [batch, 32, 7, 7]
        x = self.pool(self.relu(self.conv3(x)))  # [batch, 64, 3, 3]  (7 halves to 3: pooling rounds down)
        return self.linear(self.flatten(x))      # [batch, num_classes]


def build_model(num_classes: int = 10) -> nn.Module:
    """Construct and return the model (currently SimpleCNN)."""
    return SimpleCNN(num_classes)

