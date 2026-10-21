import torch
from PIL import Image

from dataset import get_transforms
from model import build_model


def main():
    """Predict the digit in one 28x28 image (white digit on a black background)."""
    image_path = 'inputs/digit_1.PNG'

    # Load the image as grayscale and apply the same ToTensor + Normalize used in training.
    # Resize to 28x28: the Linear layer has weights for exactly 784 (28 x 28) pixels.
    image = Image.open(image_path).convert('L').resize((28, 28))
    train_transform, val_transform = get_transforms()
    x = val_transform(image)                       # [1, 28, 28]

    # Build the model and load the trained weights saved by train.py.
    model = build_model()
    model.load_state_dict(torch.load('checkpoint.pt'))

    # The model expects a batch, so add a batch dimension: [1, 28, 28] -> [1, 1, 28, 28].
    with torch.no_grad():
        scores = model(x.unsqueeze(0))

    probabilities = scores.softmax(dim=1)[0]
    print(f"prediction: {scores.argmax(dim=1).item()}")
    for digit, p in enumerate(probabilities.tolist()):
        print(f"  {digit}: {p:.0%}")


if __name__ == '__main__':
    main()
