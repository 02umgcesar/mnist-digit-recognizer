import torch
import torch.nn as nn

from dataset import get_dataloaders
from model import build_model


def train_epoch(model, loader, criterion, optimizer):
    """Train the model for one pass (epoch) over all batches in loader."""
    for images, labels in loader:
        # 1. predict: 10 scores per image, one per digit.
        #    A higher score means the model leans more toward that digit.
        #    Each score = sum of (pixel x that digit's weight for the pixel) + that digit's bias.
        #    Example with 4 pixels instead of 784, for digit 0:
        #      pixels  [2.8, -0.4, 2.8, -0.4]
        #      weights [0.5, -1.0, 0.3,  0.2], bias 0.1
        #      score = 2.8*0.5 + (-0.4)*(-1.0) + 2.8*0.3 + (-0.4)*0.2 + 0.1 = 2.66
        scores = model(images)

        # 2. measure: how wrong the scores are (the loss).
        #    High probability on the correct digit -> low loss -> good.
        #    Low probability on the correct digit  -> high loss -> bad.
        #    Example continuing from above, with 3 digits instead of 10, for an image whose label is 0:
        #      scores         [2.66,  1.0,  0.1]    (2.66 is digit 0's score from step 1)
        #      e^score        [14.30, 2.72, 1.11], total 18.12
        #      probabilities  [0.79,  0.15, 0.06]   (each e^score / total; they add up to 1)
        #      loss = -log(probability of digit 0) = -log(0.79) = 0.24
        loss = criterion(scores, labels)

        # 3. reset every weight's gradient to 0 before computing new ones.
        #    backward() ADDS to the stored gradients instead of replacing them,
        #    so without this, the last batch's gradients would mix into this batch's.
        #    Example for one weight: last batch's gradient -0.5, this batch's gradient +0.2
        #      with zero_grad:    step() uses +0.2               (correct: this batch only)
        #      without zero_grad: step() uses -0.5 + 0.2 = -0.3  (wrong direction)
        #    In the first batch there are no gradients yet, so this does nothing.
        optimizer.zero_grad()

        # 4. compute a gradient for every weight (7,840 weights + 10 biases).
        #    Each weight's gradient = average over the 64 images in the batch of (probability - target) x pixel,
        #    where target is 1 if the image's label is that weight's digit, otherwise 0.
        #    Negative gradient -> the weight should go up; positive -> it should go down.
        #      Weight goes up:   a bright pixel there will push that digit's score higher (counts FOR the digit).
        #      Weight goes down: a bright pixel there will push that digit's score lower (counts AGAINST the digit).
        #    Example, continuing from above: digit 0's weight at pixel 1, in a batch of 3 images instead of 64:
        #      image A, label 0: (0.79 - 1) x  2.8 = -0.59   (the image from above; target 1)
        #      image B, label 1: (0.10 - 0) x -0.4 = -0.04   (not a 0; target 0)
        #      image C, label 7: (0.05 - 0) x  2.8 = +0.14   (not a 0; target 0)
        #      gradient = (-0.59 - 0.04 + 0.14) / 3 = -0.16  -> raise this weight
        loss.backward()

        # 5. update every weight using its gradient. This is the only line that changes the model.
        #    Basic rule: new weight = old weight - learning rate x gradient
        #    So each weight moves opposite to its gradient (negative -> up, positive -> down).
        #    Example, continuing from above: weight 0.5, gradient -0.16, learning rate 0.001:
        #      0.5 - 0.001 x (-0.16) = 0.50016  -> a small step up
        #    Adam adjusts each weight's step size, but always moves it in this direction.
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
    # model.state_dict() is a dictionary of the model's learned values:
    #   'linear.weight' -> [10, 784] tensor, 'linear.bias' -> [10] tensor
    torch.save(model.state_dict(), 'checkpoint.pt')
    # print("saved weights to checkpoint.pt")


if __name__ == '__main__':
    main()
