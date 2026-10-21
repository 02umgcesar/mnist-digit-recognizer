import torch
import torch.nn as nn

from dataset import get_dataloaders
from model import build_model


def train_epoch(model, loader, criterion, optimizer):
    """Train the model for one pass (epoch) over all batches in loader."""
    for images, labels in loader:
        # 1. Predict: run the batch through the model to get 10 scores per image (higher = more likely that digit).
        scores = model(images)

        # 2. Measure: compare the scores with the labels to get one loss number (lower = better).
        loss = criterion(scores, labels)

        # 3. Clear: reset the last batch's gradients, since backward() adds to them instead of replacing them.
        optimizer.zero_grad()

        # 4. Gradients: work out which way, and how much, each weight should move to lower the loss.
        loss.backward()

        # 5. Update: move every weight a small step in that direction. This is the only line that changes the model.
        optimizer.step()

        # Uncomment to see the loss every 100 batches.
        # if batch_num % 100 == 0:
            # print(f"batch {batch_num}: loss {loss.item():.3f}")


def validate(model, loader):
    """Return the fraction of images in loader that the model gets right (its accuracy)."""
    correct = 0
    total = 0

    # No learning happens here, so tell PyTorch not to record calculations for backward().
    with torch.no_grad():
        for images, labels in loader:
            scores = model(images)

            # The predicted digit is the position of the highest of the 10 scores.
            #   e.g. scores [2.66, 1.0, 0.1] -> prediction 0
            predictions = scores.argmax(dim=1)

            # Count how many predictions match the labels.
            #   e.g. predictions [0, 3, 7], labels [0, 1, 7] -> [True, False, True] -> 2 correct
            correct += (predictions == labels).sum().item()
            total += len(labels)

    return correct / total


def main():
    """Set up everything training needs: data, model, loss function, optimizer."""
    epochs = 5
    batch_size = 64
    learning_rate = 0.001

    train_loader, val_loader = get_dataloaders(batch_size=batch_size)
    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Each epoch is one full pass over all 60,000 training images.
    # The same model keeps learning, so each epoch starts from where the last one ended.
    for epoch in range(0, epochs):
        train_epoch(model, train_loader, criterion, optimizer)

        # Uncomment to see the validation accuracy after each epoch.    
        # accuracy = validate(model, val_loader)
        # print(f"epoch {epoch+1}/{epochs}: validation accuracy {accuracy:.1%}")

    # Save the trained weights to a file, so they aren't lost when the script ends.
    # model.state_dict() is a dictionary of the model's learned values, named after the layers in model.py:
    #   'conv1.weight' -> [16, 1, 3, 3] (16 filters of 3x3), 'conv1.bias' -> [16]
    #   'conv2.weight' -> [32, 16, 3, 3], 'conv3.weight' -> [64, 32, 3, 3] (plus their biases)
    #   'linear.weight' -> [10, 576], 'linear.bias' -> [10]
    torch.save(model.state_dict(), 'checkpoint.pt')
    # print("saved weights to checkpoint.pt")


if __name__ == '__main__':
    main()
