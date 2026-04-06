# Neural Network with PyTorch

A clean, professional template for regression tasks using feedforward neural networks in PyTorch.

## Project Structure

```
nn-pytorch/
├── scripts/
│   ├── train.py         # training entry point
│   └── evaluate.py      # evaluation entry point
├── src/nn_pytorch/
│   ├── config.py        # paths and hyperparameters
│   ├── data.py          # data loading, splitting, scaling
│   ├── trainer.py       # training and evaluation loops
│   ├── plots.py         # loss and parity plots
│   ├── models/
│   │   └── FNN.py       # feedforward neural network
│   └── utils/
│       └── device.py    # device detection
├── data/                # raw data (not tracked)
├── models/              # saved checkpoints (not tracked)
├── figures/             # plots and outputs (not tracked)
├── notebooks/           # exploratory notebooks
├── pyproject.toml       # dependencies
└── uv.lock              # locked dependencies
```

## Setup

**Requirements:** Python 3.11+, [uv](https://github.com/astral-sh/uv)
```bash
# clone
git clone https://github.com/findmussa/NN_pytorch.git
cd NN_pytorch

# install dependencies
uv sync
```

## Usage

**1. Add your data**
```bash
# place your CSV file at
data/data.csv
```

**2. Configure**

Edit `src/nn_pytorch/config.py` to set hyperparameters:
```python
EPOCHS       = 500
BATCH_SIZE   = 32
LR           = 1e-3
PATIENCE     = 20
H1           = 64
H2           = 32
RANDOM_STATE = 1
```

**3. Train**
```bash
uv run python scripts/train.py
```

**4. Evaluate**
```bash
uv run python scripts/evaluate.py
```

## Model

Feedforward Neural Network (FNN) with two hidden layers:
Input → Linear → ReLU → Linear → ReLU → Linear → Output
H1                H2

## Dependencies

| Package | Purpose |
|---|---|
| PyTorch | deep learning framework |
| NumPy | numerical computing |
| Pandas | data loading |
| scikit-learn | preprocessing, metrics |
| Matplotlib | plotting |
| torchinfo | model summary |

## Results

After training, results are saved to:
- `models/checkpoint.pth` — best model weights + scalers
- `figures/loss.pdf` — training and validation loss curves
- `figures/parity.pdf` — predicted vs true values

## Author

**Nur MM Kalimullah, PhD** — Research Fellow, Trinity College Dublin  
GitHub: [@findmussa](https://github.com/findmussa)  
Google Scholar: [Nur MM Kalimullah](https://scholar.google.com/citations?user=yrrCtqwAAAAJ&hl=en)
Web: (https://findmussa.github.io)