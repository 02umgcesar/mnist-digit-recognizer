# MNIST Digit Recognizer

A handwritten digit classifier built with PyTorch, trained on the MNIST dataset and tested on my own handwriting.

Built as a learning project to understand `Dataset`, `DataLoader`, batching, and the difference between training and test performance.

## Setup

Requires Python 3.12. Create a virtual environment in the project root and install the dependencies into it:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Run `source .venv/bin/activate` in each new terminal before working on the project.

**VS Code:** run **Python: Select Interpreter** and choose `.venv/bin/python`. If Pylance reports `Import "torch" could not be resolved`, VS Code is using a different Python. Select the `.venv` interpreter again.
