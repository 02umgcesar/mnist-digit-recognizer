# MNIST Digit Recognizer

A handwritten digit classifier built with PyTorch, trained on the MNIST dataset and tested on my own handwriting.

## Setup

Requires Python 3.12.

```bash
python3 -m pip install -r requirements.txt
```

## Usage

Run all commands from the project root.

**1. Train the model** (once). This downloads MNIST into `data/`, trains for 5 epochs, and saves the weights to `checkpoint.pt`:

```bash
python3 src/train.py
```

**2. Predict a digit.** Set `image_path` in `src/predict.py` to your image, then run:

```bash
python3 src/predict.py
```

The image should show a single white digit on a black background. Any size works; it is resized to 28 × 28.
