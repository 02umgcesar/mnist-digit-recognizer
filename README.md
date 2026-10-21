# MNIST Digit Recognizer

A handwritten digit classifier built with PyTorch: a small convolutional neural network (CNN) trained on the MNIST dataset and tested on my own handwriting.

## Setup

Requires Python 3.12.

```bash
python3 -m pip install -r requirements.txt
```

## Usage

Run all commands from the project root.

**1. Train the model.** This downloads MNIST into `data/`, trains for 5 epochs, and saves the weights to `checkpoint.pt`. Run it again whenever you change the model in `src/model.py`.

```bash
python3 src/train.py
```

**2. Draw a digit** (optional). This opens a drawing pad in your browser. Draw a digit and click **Save to inputs/** to save it as a 28 × 28 image in the MNIST format. Press Ctrl+C in the terminal to stop the pad.

```bash
python3 src/digit_pad.py
```

**3. Predict a digit.** Set `image_path` in `src/predict.py` to your image (for example `inputs/digit.png`), then run:

```bash
python3 src/predict.py
```

The image should show a single white digit on a black background. Any size works; it is resized to 28 × 28.

## How the CNN works

Open `cnn_block_explorer.html` in a web browser for an interactive walkthrough of one convolution block: a 3 × 3 filter sliding over a digit with padding, then ReLU, then max pooling.
