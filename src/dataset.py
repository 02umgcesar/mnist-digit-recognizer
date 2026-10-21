from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Mean and std of MNIST training pixels (after scaling to [0, 1]).
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Return (train_transform, val_transform).

    ToTensor turns a 28x28 PIL image (pixels 0-255) into a float tensor of
    shape [1, 28, 28] with values in [0, 1]. Normalize then centers the values
    around 0, which makes training more stable. Both pipelines are identical
    for now; data augmentation would go in the train pipeline only.
    """
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])
    return train_transform, val_transform


def download_dataset_if_needed(data_dir: str = './data') -> None:
    """Ensure the MNIST train and test files exist in data_dir.

    torchvision skips the download if the files are already there.
    """
    datasets.MNIST(data_dir, train=True, download=True)
    datasets.MNIST(data_dir, train=False, download=True)


def get_dataloaders(data_dir: str = './data', batch_size: int = 64, num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader).

    The Dataset knows how to fetch one (image, label) pair by index. The
    DataLoader wraps it to produce shuffled batches of shape
    [batch_size, 1, 28, 28] for images and [batch_size] for labels.
    Validation uses MNIST's separate 10,000-image test split.
    """
    download_dataset_if_needed(data_dir)
    train_transform, val_transform = get_transforms()

    train_set = datasets.MNIST(data_dir, train=True, transform=train_transform)
    val_set = datasets.MNIST(data_dir, train=False, transform=val_transform)

    # Shuffle training data so each epoch sees batches in a new order.
    # Validation order doesn't affect the result, so keep it fixed.
    #   | Workers | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Average |
    #   |---------|-------|-------|-------|-------|-------|---------|
    #   | 0       | 2.3   | 2.3   | 2.6   | 2.3   | 2.9   | 2.5     |
    #   | 1       | 4.2   | 4.2   | 4.6   | 4.2   | 4.7   | 4.4     |
    #   | 2       | 3.7   | 4.0   | 3.6   | 3.6   | 3.6   | 3.7     |
    #   | 3       | 3.8   | 4.0   | 4.3   | 3.8   | 4.5   | 4.1     |
    #   | 4       | 4.2   | 4.2   | 5.3   | 4.7   | 5.0   | 4.7     |

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


if __name__ == '__main__':
    # Sanity check: load one batch and inspect it.
    train_loader, val_loader = get_dataloaders()

    # Prints 60,000 in 938 batches -- 64 images/batch. Uncomment to see it.
    # print(f"train: {len(train_loader.dataset)} images in {len(train_loader)} batches")
    # Prints 10,000 in 157 batches -- 64 images/batch
    # print(f"val:   {len(val_loader.dataset)} images in {len(val_loader)} batches")


    # Data about the first batch of images and labels. Uncomment to see it.
    # images, labels = next(iter(train_loader))
    # print(f"images: shape={tuple(images.shape)} dtype={images.dtype} "
        #   f"min={images.min():.2f} max={images.max():.2f} mean={images.mean():.2f}")
    # print(f"labels: shape={tuple(labels.shape)} dtype={labels.dtype} first 10={labels[:10].tolist()}")
